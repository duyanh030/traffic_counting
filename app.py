import json
import math
from pathlib import Path
import streamlit as st
from PIL import Image

import config
from ultralytics import YOLO
from view_heatmap import generate_heatmap_for_image
from track_loss_diagnoser import diagnose_candidate_frame
from cvat_integration import (
    push_approved_to_cvat,
    export_approved_to_zip,
    get_approved_candidates,
)

st.set_page_config(
    page_title="Traffic Counting - Active Learning Studio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_cached_yolo(model_name: str = "yolov8n.pt"):
    """Cache YOLO model in memory so heatmap generation is instantaneous."""
    return YOLO(model_name)


def load_runs_data():
    """Discover all model versions and execution runs (including legacy root runs)."""
    base_dir = config.CANDIDATES_DIR
    runs_tree = {}

    if not base_dir.exists():
        return runs_tree

    # 1. Backwards compatibility: Check for candidates directly in root (candidates/metadata)
    root_meta = base_dir / "metadata"
    if root_meta.is_dir() and any(root_meta.glob("*.json")):
        runs_tree.setdefault("yolov8n", []).append(
            {
                "path": base_dir,
                "name": "run_initial_candidates (Root)",
                "report": {
                    "source_name": "Previous Video Run",
                    "start_time": "Initial Session",
                    "counting_summary": {"vehicles_in": 0, "vehicles_out": 419},
                    "total_frames_processed": 14640,
                },
            }
        )

    # 2. Check for nested directories: candidates/<model_dir>/run_*
    for model_dir in sorted(base_dir.iterdir()):
        if (
            model_dir.is_dir()
            and not model_dir.name.startswith(".")
            and model_dir.name not in ["images", "previews", "metadata"]
        ):
            runs = []
            for run_dir in sorted(model_dir.iterdir(), reverse=True):
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    report_path = run_dir / "run_report.json"
                    report = {}
                    if report_path.is_file():
                        try:
                            with open(report_path, "r", encoding="utf-8") as f:
                                report = json.load(f)
                        except Exception:
                            pass
                    runs.append({"path": run_dir, "name": run_dir.name, "report": report})
            if runs:
                runs_tree[model_dir.name] = runs

    return runs_tree


@st.cache_data(show_spinner="Loading candidate metadata...")
def load_candidates_from_run(run_dir_str: str):
    """Load all candidate metadata JSON files from a specific run directory."""
    run_dir = Path(run_dir_str)
    meta_dir = run_dir / "metadata"
    candidates = []

    if not meta_dir.exists():
        return candidates

    for meta_file in sorted(meta_dir.glob("*.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_meta_path"] = str(meta_file)
                data["_run_dir"] = str(run_dir)
                candidates.append(data)
        except Exception:
            continue

    return candidates


def update_candidate_status(meta_path: str, new_status: str):
    """Update review status in the metadata JSON file and clear cache."""
    try:
        path = Path(meta_path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["status"] = new_status
            data["reviewed"] = True
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            st.cache_data.clear()
            return True
    except Exception as e:
        st.error(f"Error updating metadata: {e}")
    return False


def batch_update_status(candidates_list, new_status: str):
    """Update status for multiple candidates at once."""
    for c in candidates_list:
        try:
            path = Path(c["_meta_path"])
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["status"] = new_status
                data["reviewed"] = True
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            continue
    st.cache_data.clear()


def main():
    st.title("🚗 Traffic Counting — Active Learning Review Studio")
    st.caption("Review suspicious candidates (track-loss & low-confidence) and push directly to CVAT.")

    runs_tree = load_runs_data()

    if not runs_tree:
        st.warning(
            f"No candidate runs found in `{config.CANDIDATES_DIR}`. "
            "Please run `python count_traffic.py --source <video_path>` first to generate candidates!"
        )
        return

    # --- SIDEBAR: RUN SELECTION ---
    st.sidebar.header("📁 Session Selector")
    model_list = list(runs_tree.keys())
    selected_model = st.sidebar.selectbox("Select Model Version", model_list)

    available_runs = runs_tree[selected_model]
    run_options = {run["name"]: run for run in available_runs}
    selected_run_name = st.sidebar.selectbox("Select Execution Run", list(run_options.keys()))
    selected_run = run_options[selected_run_name]
    run_dir = selected_run["path"]
    report = selected_run.get("report", {})

    # Display Run Overview in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Session Metrics")
    if report:
        st.sidebar.write(f"**Source:** `{report.get('source_name', 'Unknown')}`")
        st.sidebar.write(f"**Start Time:** `{report.get('start_time', 'N/A')}`")
        st.sidebar.write(f"**Duration:** `{report.get('duration_seconds', 0)}s`")
        st.sidebar.write(f"**Frames Processed:** `{report.get('total_frames_processed', 0)}`")
        cnt = report.get("counting_summary", {})
        st.sidebar.write(f"**Vehicles IN / OUT:** `{cnt.get('vehicles_in', 0)} / {cnt.get('vehicles_out', 0)}`")

    # --- LOAD CANDIDATES ---
    candidates = load_candidates_from_run(str(run_dir))
    total_cand = len(candidates)

    if total_cand == 0:
        st.info("No candidates were captured in this run.")
        return

    # Count statuses
    pending_count = sum(1 for c in candidates if c.get("status", "pending") == "pending")
    approved_count = sum(1 for c in candidates if c.get("status") == "approved")
    rejected_count = sum(1 for c in candidates if c.get("status") == "rejected")

    # --- METRICS BANNER ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Candidates", total_cand)
    col_m2.metric(
        "Pending Review",
        pending_count,
        delta=f"{pending_count/total_cand*100:.1f}%" if total_cand else None,
    )
    col_m3.metric("Approved for CVAT", approved_count, delta=f"+{approved_count}" if approved_count else None)
    col_m4.metric("Rejected (False Alarm)", rejected_count)

    st.markdown("---")

    # --- FILTERS & BATCH ACTIONS ---
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 3])

    with f_col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "pending", "approved", "rejected"],
            index=1,  # Default to 'pending'
        )

    with f_col2:
        reason_filter = st.selectbox(
            "Filter by Reason",
            ["All", "track_loss", "low_confidence"],
        )

    with f_col3:
        all_classes = sorted(list({c.get("details", {}).get("class_name", "unknown") for c in candidates}))
        class_filter = st.selectbox("Filter by Class", ["All"] + all_classes)

    # Filtered candidate list
    filtered = []
    for c in candidates:
        if status_filter != "All" and c.get("status", "pending") != status_filter:
            continue
        if reason_filter != "All" and c.get("reason") != reason_filter:
            continue
        if class_filter != "All" and c.get("details", {}).get("class_name") != class_filter:
            continue
        filtered.append(c)

    with f_col4:
        st.write(f"**Showing {len(filtered)} / {total_cand} candidates**")
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("✅ Approve All Filtered", use_container_width=True):
            batch_update_status(filtered, "approved")
            st.rerun()
        if b_col2.button("❌ Reject All Filtered", use_container_width=True):
            batch_update_status(filtered, "rejected")
            st.rerun()

    # --- MAIN CONTENT TABS ---
    tab_gallery, tab_inspector, tab_cvat, tab_summary = st.tabs(
        ["🖼️ Gallery View", "🔍 Detail Inspector", "🚀 Push to CVAT", "📈 Run History & Reports"]
    )

    # --- TAB 1: GALLERY VIEW (WITH PAGINATION) ---
    with tab_gallery:
        if not filtered:
            st.info("No candidates match the selected filters.")
        else:
            page_size = 24
            total_pages = max(1, math.ceil(len(filtered) / page_size))

            pg_col1, pg_col2, pg_col3 = st.columns([1, 2, 1])
            with pg_col2:
                current_page = st.number_input(
                    f"Page (1 to {total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                )

            start_idx = (current_page - 1) * page_size
            end_idx = min(start_idx + page_size, len(filtered))
            page_items = filtered[start_idx:end_idx]

            st.caption(f"Displaying candidates {start_idx + 1} - {end_idx} of {len(filtered)}")

            cols_per_row = 3
            for i in range(0, len(page_items), cols_per_row):
                row_candidates = page_items[i : i + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, item in zip(cols, row_candidates):
                    with col:
                        cand_id = item["candidate_id"]
                        reason = item.get("reason", "unknown")
                        frame_idx = item.get("frame_idx", 0)
                        status = item.get("status", "pending")
                        details = item.get("details", {})
                        cls_name = details.get("class_name", "vehicle")
                        conf = details.get("confidence", details.get("last_confidence", 0.0))

                        # Status badge styling
                        badge_color = {
                            "pending": "orange",
                            "approved": "green",
                            "rejected": "gray",
                        }.get(status, "blue")
                        st.markdown(
                            f"**Frame #{frame_idx}** | `:{badge_color}[{status.upper()}]` | `{reason}`"
                        )

                        # Load preview image (annotated)
                        preview_path = run_dir / item.get("preview_file", "")
                        if preview_path.is_file():
                            img = Image.open(preview_path)
                            st.image(img, use_container_width=True)
                        else:
                            raw_path = run_dir / item.get("image_file", "")
                            if raw_path.is_file():
                                st.image(Image.open(raw_path), use_container_width=True)
                            else:
                                st.warning("Image file not found.")

                        st.caption(f"**Class:** {cls_name} | **Conf:** {conf:.2f}")

                        # Action buttons
                        btn_c1, btn_c2 = st.columns(2)
                        if btn_c1.button("✅ Approve", key=f"app_{cand_id}", use_container_width=True):
                            update_candidate_status(item["_meta_path"], "approved")
                            st.rerun()
                        if btn_c2.button("❌ Reject", key=f"rej_{cand_id}", use_container_width=True):
                            update_candidate_status(item["_meta_path"], "rejected")
                            st.rerun()

                        st.markdown("---")

    # --- TAB 2: DETAIL INSPECTOR ---
    with tab_inspector:
        if not filtered:
            st.info("No candidates available for inspection.")
        else:
            cand_names = [
                f"Frame {c.get('frame_idx')} - {c.get('candidate_id')} ({c.get('status')})"
                for c in filtered[:200]
            ]
            selected_idx = st.selectbox(
                "Select Candidate to Inspect",
                range(len(cand_names)),
                format_func=lambda i: cand_names[i],
            )
            inspect_item = filtered[selected_idx]

            raw_path = run_dir / inspect_item.get("image_file", "")
            prev_path = run_dir / inspect_item.get("preview_file", "")

            insp_col1, insp_col2 = st.columns(2)
            with insp_col1:
                st.subheader("Raw Frame (For CVAT Annotation)")
                if raw_path.is_file():
                    st.image(Image.open(raw_path), use_container_width=True)
                else:
                    st.error("Raw image not found.")

            with insp_col2:
                st.subheader("Preview Frame (AI Alert Highlight)")
                if prev_path.is_file():
                    st.image(Image.open(prev_path), use_container_width=True)
                else:
                    st.error("Preview image not found.")

            # --- SMART ROOT CAUSE DIAGNOSTIC ASSISTANT ---
            st.markdown("---")
            st.subheader("🩺 AI Smart Diagnostic Assistant (Bắt bệnh & Khuyên Approve/Reject)")
            diag_img_path = raw_path if raw_path.is_file() else prev_path
            if diag_img_path.is_file():
                try:
                    cached_model = get_cached_yolo(config.MODEL_NAME)
                    diag = diagnose_candidate_frame(
                        frame_input=str(diag_img_path),
                        details=inspect_item.get("details", {}),
                        model=cached_model,
                        conf_threshold=config.CONFIDENCE_THRESHOLD,
                    )

                    rec = diag.get("recommendation", "REVIEW")
                    verdict = diag.get("verdict", "UNKNOWN")

                    if rec == "APPROVE":
                        st.success(
                            f"### 🟢 KHUYÊN BẤM APPROVE: {diag.get('title')}\n\n"
                            f"**Phân tích:** {diag.get('explanation')}\n\n"
                            f"👉 **Đề xuất:** Hãy bấm **✅ Mark as Approved** bên dưới để đẩy vào CVAT gán nhãn cho đợt train tới!"
                        )
                    elif rec == "REJECT":
                        icon = "🟠" if verdict == "ID_SWITCH" else "⚪"
                        alert_fn = st.warning if verdict == "ID_SWITCH" else st.info
                        alert_fn(
                            f"### {icon} KHUYÊN BẤM REJECT: {diag.get('title')}\n\n"
                            f"**Phân tích:** {diag.get('explanation')}\n\n"
                            f"👉 **Đề xuất:** Hãy bấm **❌ Mark as Rejected** để bỏ qua, không tốn thời gian gán nhãn vô ích."
                        )
                    else:
                        st.info(f"ℹ️ **Thông tin:** {diag.get('explanation')}")
                except Exception as e:
                    st.warning(f"Không thể chạy chẩn đoán tự động: {e}")

            st.markdown("### Metadata Details")
            st.json(inspect_item)

            # --- ON-DEMAND AI ATTENTION HEATMAP (DEBUGGER) ---
            st.markdown("---")
            st.subheader("🔥 AI Attention Heatmap (Debug Feature Occlusion / Texture Blending)")
            st.caption(
                "Click to inspect where YOLO's neural network was focusing its attention on this suspicious candidate frame."
            )

            hm_col1, hm_col2 = st.columns([2, 1])
            with hm_col1:
                layer_options = {
                    15: "Layer 15 (P3 - Recommended for Motorcycles & Small Vehicles)",
                    18: "Layer 18 (P4 - Recommended for Cars / Medium Vehicles)",
                    21: "Layer 21 (P5 - Recommended for Trucks & Large Vehicles)",
                    9: "Layer 9 (SPPF - Background & Shadow Invariance)",
                }
                selected_hm_layer = st.selectbox(
                    "Target Layer to Inspect",
                    options=list(layer_options.keys()),
                    format_func=lambda l: layer_options[l],
                    index=0,  # Layer 15 by default
                    key=f"layer_select_{inspect_item.get('candidate_id', 'curr')}",
                )

            with hm_col2:
                st.write("")
                st.write("")
                gen_hm_btn = st.button(
                    "⚡ Generate Heatmap",
                    key=f"btn_hm_{inspect_item.get('candidate_id', 'curr')}",
                    use_container_width=True,
                )

            if gen_hm_btn:
                img_to_inspect = raw_path if raw_path.is_file() else prev_path
                if not img_to_inspect.is_file():
                    st.error("Image file not available for heatmap generation.")
                else:
                    with st.spinner(f"Analyzing neural network activations at Layer {selected_hm_layer}..."):
                        try:
                            cached_model = get_cached_yolo(config.MODEL_NAME)
                            _, comparison_rgb, l_name = generate_heatmap_for_image(
                                str(img_to_inspect),
                                layer_idx=selected_hm_layer,
                                model=cached_model,
                                conf=config.CONFIDENCE_THRESHOLD,
                            )
                            st.image(
                                comparison_rgb,
                                caption=f"Detection Box (Left) vs. Eigen-CAM Heatmap (Right) — Layer {selected_hm_layer}: {l_name}",
                                use_container_width=True,
                            )
                            st.info(
                                "💡 **Cách chẩn đoán nhanh:**\n"
                                "- **Đốm đỏ rực tập trung vào xe**: AI nhìn thấy đặc trưng tốt.\n"
                                "- **Vùng xe bị màu xanh lam/xám nhạt**: Bị **Texture Blending** (xe bị hòa lẫn vào mặt đường bê tông/nhựa).\n"
                                "- **Đốm đỏ bị xe lớn bên cạnh hoặc bóng đổ hút trọn**: Bị **Feature Occlusion** (xe máy bị lấn át đặc trưng)."
                            )
                        except Exception as e:
                            st.error(f"Lỗi khi tạo Heatmap: {e}")

            st.markdown("---")
            act_col1, act_col2, _ = st.columns([1, 1, 3])
            if act_col1.button("✅ Mark as Approved", key="insp_app"):
                update_candidate_status(inspect_item["_meta_path"], "approved")
                st.rerun()
            if act_col2.button("❌ Mark as Rejected", key="insp_rej"):
                update_candidate_status(inspect_item["_meta_path"], "rejected")
                st.rerun()

    # --- TAB 3: PUSH TO CVAT ---
    with tab_cvat:
        st.subheader("🚀 Export Approved Candidates to CVAT")
        st.write(
            "Push all approved high-value frames to CVAT for labeling, or export them to a local zip package."
        )

        appr_list = get_approved_candidates(run_dir)
        num_approved = len(appr_list)

        if num_approved == 0:
            st.warning(
                "⚠️ There are **0 candidates marked as 'approved'** in this session yet.\n\n"
                "Please go to **Gallery View**, select the best error frames and click **✅ Approve** before pushing to CVAT!"
            )
        else:
            st.success(f"🎯 **{num_approved} approved candidates** are ready to be pushed to CVAT!")

        st.markdown("---")
        cvat_c1, cvat_c2 = st.columns(2)

        with cvat_c1:
            st.markdown("#### 🌐 CVAT Connection Settings")
            input_cvat_url = st.text_input("CVAT Server URL", value=config.CVAT_URL)
            input_cvat_user = st.text_input("CVAT Username", value=config.CVAT_USERNAME)
            input_cvat_pass = st.text_input("CVAT Password", value=config.CVAT_PASSWORD, type="password")
            input_task_name = st.text_input(
                "Task Name",
                value=f"Traffic_ActiveLearning_{selected_run_name}",
            )
            enable_pre_annotate = st.checkbox(
                "🤖 Auto Pre-annotate with current YOLO (Auto-draws obvious vehicles to save ~85% time)",
                value=True,
            )

            if st.button("🚀 Push to CVAT Now", disabled=(num_approved == 0), type="primary", use_container_width=True):
                with st.spinner(f"Connecting to CVAT, uploading {num_approved} images and generating pre-annotations..."):
                    try:
                        res = push_approved_to_cvat(
                            run_dir=run_dir,
                            task_name=input_task_name,
                            cvat_url=input_cvat_url,
                            username=input_cvat_user,
                            password=input_cvat_pass,
                            enable_pre_annotation=enable_pre_annotate,
                        )
                        if res.get("success"):
                            st.balloons()
                            boxes_info = f" with **{res.get('pre_annotated_boxes', 0)} boxes pre-drawn by AI!**" if enable_pre_annotate else ""
                            st.success(
                                f"🎉 **Task Created Successfully!** Task ID: `#{res['task_id']}` ({res['count']} images{boxes_info})"
                            )
                            st.markdown(
                                f"### 👉 [Open Task #{res['task_id']} on CVAT]({res['task_url']})"
                            )
                        else:
                            st.error(f"Failed: {res.get('error')}")
                    except Exception as e:
                        st.error(f"❌ Error connecting to CVAT: {e}")

        with cvat_c2:
            st.markdown("#### 📦 Offline Export (ZIP)")
            st.write(
                "If CVAT is offline or hosted on a different machine, export all approved raw images into a clean zip archive for manual upload."
            )
            export_zip_path = config.BASE_DIR / "reports" / f"export_{selected_run_name}.zip"

            if st.button("📦 Export to Local ZIP Archive", disabled=(num_approved == 0), use_container_width=True):
                try:
                    count, zip_p = export_approved_to_zip(run_dir, export_zip_path)
                    st.success(f"✅ Exported {count} approved images to:\n`{zip_p}`")
                except Exception as e:
                    st.error(f"Export failed: {e}")

    # --- TAB 4: RUN HISTORY & SUMMARY ---
    with tab_summary:
        st.subheader("📜 Global Execution History")
        history_file = config.REPORTS_DIR / "runs_history.json"
        if history_file.is_file():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
                if hist_data:
                    table_rows = []
                    for h in reversed(hist_data):
                        cnt = h.get("counting_summary", {})
                        snaps = h.get("suspicious_snapshots_summary", {})
                        table_rows.append(
                            {
                                "Run ID": h.get("run_id"),
                                "Source": h.get("source_name"),
                                "Model": h.get("model_version"),
                                "Start Time": h.get("start_time"),
                                "Duration (s)": h.get("duration_seconds"),
                                "Frames": h.get("total_frames_processed"),
                                "IN": cnt.get("vehicles_in", 0),
                                "OUT": cnt.get("vehicles_out", 0),
                                "Candidates": snaps.get("total_captured", 0),
                            }
                        )
                    st.dataframe(table_rows, use_container_width=True)
                else:
                    st.info("History file is empty.")
            except Exception as e:
                st.error(f"Failed to read history: {e}")
        else:
            st.info("No global runs history found yet in `reports/runs_history.json`.")

        st.markdown("---")
        st.subheader("📄 Current Run Report JSON")
        st.json(report)


if __name__ == "__main__":
    main()
