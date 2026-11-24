import json
import math
from pathlib import Path
import streamlit as st
from PIL import Image

import config
from ultralytics import YOLO
from view_heatmap import generate_heatmap_for_image
from track_loss_diagnoser import diagnose_candidate_frame
from model_deployer import (
    get_active_model_info,
    list_staged_models,
    approve_and_deploy_model,
    reject_staged_model,
    revert_to_version,
)
from cvat_integration import (
    push_approved_to_cvat,
    export_approved_to_zip,
    get_approved_candidates,
)
from golden_test_manager import (
    add_candidate_to_golden_set,
    load_golden_cases,
    run_regression_test,
    draw_golden_case_preview,
    get_golden_case_image_path,
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
    active_model_info = get_active_model_info()
    current_active_ver = active_model_info.get("active_version", "v1.0.0")
    current_artifact_name = Path(active_model_info.get("model_path", "yolov8n.pt")).name

    model_list = list(runs_tree.keys())

    def format_model_label(m: str) -> str:
        if m == "yolov8_traffic_latest":
            return f"{m} (Active: {current_active_ver})"
        return m

    selected_model = st.sidebar.selectbox(
        "Select Model Version",
        model_list,
        format_func=format_model_label,
    )

    available_runs = runs_tree[selected_model]
    run_options = {run["name"]: run for run in available_runs}
    selected_run_name = st.sidebar.selectbox("Select Execution Run", list(run_options.keys()))
    selected_run = run_options[selected_run_name]
    run_dir = selected_run["path"]
    report = selected_run.get("report", {})

    # Extract model lineage for this session run
    run_sem_ver = report.get("semantic_version")
    if not run_sem_ver:
        if selected_model == "yolov8_traffic_latest":
            run_sem_ver = current_active_ver
        else:
            run_sem_ver = selected_model
    run_artifact_name = report.get("model_artifact", current_artifact_name)

    # Display Run Overview in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Session Metrics")
    if report:
        st.sidebar.write(f"**Model Version:** `{run_sem_ver} ({run_artifact_name})`")
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
        if b_col1.button("✅ Approve All Filtered", width="stretch"):
            batch_update_status(filtered, "approved")
            st.rerun()
        if b_col2.button("❌ Reject All Filtered", width="stretch"):
            batch_update_status(filtered, "rejected")
            st.rerun()

    # --- MAIN CONTENT TABS ---
    tab_gallery, tab_inspector, tab_cvat, tab_deploy, tab_summary = st.tabs(
        ["🖼️ Gallery View", "🔍 Detail Inspector", "🚀 Push to CVAT", "🎯 Model Deployment", "📈 Run History & Reports"]
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

            st.info(
                f"📌 Showing candidates captured by Model Version: **`{run_sem_ver}`** *(Artifact: `{run_artifact_name}`)*"
            )
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
                        cand_ver = item.get("model_version", run_sem_ver)

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
                            st.image(img, width="stretch")
                        else:
                            raw_path = run_dir / item.get("image_file", "")
                            if raw_path.is_file():
                                st.image(Image.open(raw_path), width="stretch")
                            else:
                                st.warning("Image file not found.")

                        st.caption(f"**Model:** `{cand_ver}` | **Class:** {cls_name} | **Conf:** {conf:.2f}")

                        # Action buttons
                        btn_c1, btn_c2 = st.columns(2)
                        if btn_c1.button("✅ Approve", key=f"app_{cand_id}", width="stretch"):
                            update_candidate_status(item["_meta_path"], "approved")
                            st.rerun()
                        if btn_c2.button("❌ Reject", key=f"rej_{cand_id}", width="stretch"):
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
            cand_ver = inspect_item.get("model_version", run_sem_ver)
            cand_art = inspect_item.get("model_artifact", run_artifact_name)
            st.markdown(f"#### 🏷️ Candidate Lineage: `{inspect_item.get('candidate_id')}`")
            st.caption(
                f"Captured by Model Version: **`{cand_ver}`** | "
                f"Weights Artifact: **`{cand_art}`** | "
                f"Frame: **`#{inspect_item.get('frame_idx')}`**"
            )

            raw_path = run_dir / inspect_item.get("image_file", "")
            prev_path = run_dir / inspect_item.get("preview_file", "")
            target_img_path = raw_path if raw_path.is_file() else prev_path

            insp_col1, insp_col2 = st.columns(2)
            with insp_col1:
                st.subheader("Raw Frame (For CVAT Annotation)")
                if raw_path.is_file():
                    st.image(Image.open(raw_path), width="stretch")
                else:
                    st.error("Raw image not found.")

            with insp_col2:
                st.subheader("Preview Frame (AI Alert Highlight)")
                if prev_path.is_file():
                    st.image(Image.open(prev_path), width="stretch")
                else:
                    st.error("Preview image not found.")

            # --- SMART ROOT CAUSE DIAGNOSTIC ASSISTANT ---
            st.markdown("---")
            st.subheader("🩺 AI Smart Diagnostic Assistant (Root Cause & Action Recommendation)")
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
                            f"### 🟢 RECOMMENDATION: APPROVE — {diag.get('title')}\n\n"
                            f"**Analysis:** {diag.get('explanation')}\n\n"
                            f"👉 **Action:** Click **✅ Mark as Approved** below to push this frame to CVAT for the next retraining round!"
                        )
                    elif rec == "REJECT":
                        icon = "🟠" if verdict == "ID_SWITCH" else "⚪"
                        alert_fn = st.warning if verdict == "ID_SWITCH" else st.info
                        alert_fn(
                            f"### {icon} RECOMMENDATION: REJECT — {diag.get('title')}\n\n"
                            f"**Analysis:** {diag.get('explanation')}\n\n"
                            f"👉 **Action:** Click **❌ Mark as Rejected** to discard this candidate. Do not waste annotation time."
                        )
                    else:
                        st.info(f"ℹ️ **Info:** {diag.get('explanation')}")
                except Exception as e:
                    st.warning(f"Could not execute automated diagnosis: {e}")

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
                    width="stretch",
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
                                width="stretch",
                            )
                            st.info(
                                "💡 **Diagnostic Quick Guide:**\n"
                                "- **Bright Red/Yellow focused on vehicle**: Neural network extracted features properly.\n"
                                "- **Cold Blue/Uniform across road**: **Texture Blending** (vehicle blended into asphalt/concrete texture).\n"
                                "- **Red hotspot absorbed into adjacent large vehicle/shadow**: **Feature Occlusion** (small vehicle features dominated by large neighbor)."
                            )
                        except Exception as e:
                            st.error(f"Error generating heatmap: {e}")

            st.markdown("---")
            act_col1, act_col2, act_col3 = st.columns([1, 1, 2])
            if act_col1.button("✅ Mark as Approved", key="insp_app", width="stretch"):
                update_candidate_status(inspect_item["_meta_path"], "approved")
                st.rerun()
            if act_col2.button("❌ Mark as Rejected", key="insp_rej", width="stretch"):
                update_candidate_status(inspect_item["_meta_path"], "rejected")
                st.rerun()
            if act_col3.button("⭐ Save to Golden Test Set (Regression Test)", key="insp_golden", width="stretch"):
                try:
                    g_res = add_candidate_to_golden_set(
                        inspect_item["_meta_path"],
                        image_path=target_img_path,
                        note=f"Frame #{inspect_item.get('frame_idx')} - {inspect_item.get('reason')} ({diag.get('verdict') if 'diag' in locals() else 'manual'})",
                    )
                    st.success(f"⭐ Saved to Golden Test Set as permanent test case `{g_res['case_id']}`!")
                except Exception as e:
                    st.error(f"Failed to add to Golden Test Set: {e}")

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

            if st.button("🚀 Push to CVAT Now", disabled=(num_approved == 0), type="primary", width="stretch"):
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

            if st.button("📦 Export to Local ZIP Archive", disabled=(num_approved == 0), width="stretch"):
                try:
                    count, zip_p = export_approved_to_zip(run_dir, export_zip_path)
                    st.success(f"✅ Exported {count} approved images to:\n`{zip_p}`")
                except Exception as e:
                    st.error(f"Export failed: {e}")

    # --- TAB 4: MODEL DEPLOYMENT & GOVERNANCE GATE ---
    with tab_deploy:
        st.subheader("🎯 Model Deployment & Governance Gate")
        st.caption(
            "Review newly retrained candidate models, evaluate validation metrics against the active production baseline, and approve promotion to production."
        )

        if "deploy_success_msg" in st.session_state:
            st.success(st.session_state.pop("deploy_success_msg"))
            st.balloons()
        if "revert_success_msg" in st.session_state:
            st.success(st.session_state.pop("revert_success_msg"))
            st.balloons()

        # 1. Active Production Model Baseline
        active_info = get_active_model_info()
        active_ver = active_info.get("active_version", "v1.0.0")
        active_model_file = active_info.get("model_path", config.MODEL_NAME)
        
        # Match active version in history rather than naive history[-1]
        active_history_entry = next(
            (h for h in active_info.get("history", []) if h.get("version") == active_ver), None
        )
        if active_history_entry:
            active_metrics = active_history_entry.get("metrics", {})
            last_deployed = active_history_entry.get("deployed_at", "Initial")
        else:
            active_metrics = active_info.get("history", [{}])[-1].get("metrics", {}) if active_info.get("history") else {}
            last_deployed = "Initial"
        active_map50 = active_metrics.get("mAP50", 0.65)

        st.markdown("#### 🏆 Active Production Model")
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        b_c1.metric("Active Version", active_ver)
        b_c2.metric("Base Model Artifact", Path(active_model_file).name)
        b_c3.metric(
            "Baseline mAP@0.5",
            f"{active_map50:.4f}" if isinstance(active_map50, (int, float)) else str(active_map50),
        )
        b_c4.metric(
            "Last Deployed",
            last_deployed,
        )

        st.markdown("---")

        # 2. Staged Candidate Models Awaiting Approval
        st.markdown("#### ⏳ Staged Model Candidates Awaiting Your Approval")
        staged_models = list_staged_models()

        if not staged_models:
            st.info(
                "✅ **No candidate models currently awaiting approval.**\n\nWhen you run `python retrain.py --epochs 30`, trained candidate models will appear here for your review and deployment."
            )
        else:
            for cand in staged_models:
                c_id = cand.get("candidate_id")
                c_metrics = cand.get("metrics", {})
                cand_map50 = c_metrics.get("mAP50", 0.0)
                cand_p = c_metrics.get("precision", 0.0)
                cand_r = c_metrics.get("recall", 0.0)
                cand_samples = cand.get("training_samples", 0)
                cand_epochs = cand.get("epochs", 30)
                cand_folder = cand.get("_folder_path")

                delta_map = cand_map50 - active_map50 if isinstance(active_map50, (int, float)) else None

                with st.container():
                    st.markdown(f"### 📦 Candidate: `{c_id}` *(Trained: {cand.get('trained_at')})*")

                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric(
                        "Validation mAP@0.5",
                        f"{cand_map50:.4f}",
                        delta=f"{delta_map:+.4f}" if delta_map is not None else None,
                    )
                    m_col2.metric("Precision", f"{cand_p:.4f}")
                    m_col3.metric("Recall", f"{cand_r:.4f}")
                    m_col4.metric("Training Samples", f"{cand_samples} ({cand_epochs} epochs)")

                    cand_manifest = cand.get("dataset_manifest")
                    if cand_manifest:
                        z_names = [z.get("archive_name") for z in cand_manifest.get("processed_zips", [])]
                        val_count = len(cand_manifest.get("val_samples", []))
                        st.caption(
                            f"📦 **Source Exports ({len(z_names)}):** `{', '.join(z_names)}` | "
                            f"📁 **Validation Dataset Split:** `{val_count} samples`"
                        )

                    # --- Golden Test Set Regression Report ---
                    reg_report = cand.get("regression_report")
                    if not reg_report:
                        cand_weights = Path(cand_folder) / "best.pt"
                        base_pt = Path(active_model_file) if Path(active_model_file).is_file() else (config.BASE_DIR / active_model_file)
                        if cand_weights.is_file():
                            reg_report = run_regression_test(
                                candidate_model_path=cand_weights,
                                baseline_model_path=base_pt if base_pt.is_file() else None,
                            )

                    if reg_report and reg_report.get("total_cases", 0) > 0:
                        st.markdown("##### 🧪 Golden Test Set Regression Benchmark")
                        p_rate = reg_report.get("pass_rate", 0.0)
                        p_count = reg_report.get("passed", 0)
                        t_count = reg_report.get("total_cases", 0)
                        status_color = "green" if p_count == t_count else "orange" if p_count > 0 else "red"

                        st.markdown(
                            f"Regression Quality: :{status_color}[**{p_count}/{t_count} Passed ({p_rate}%)**] *(Confidence >= {config.CONFIDENCE_THRESHOLD})*"
                        )

                        reg_table = []
                        for r in reg_report.get("results", []):
                            reg_table.append({
                                "Status": "✅ PASS" if r["status"] == "PASS" else "❌ FAIL",
                                "Case ID": r["case_id"],
                                "Class": r["target_class"],
                                "Baseline Conf": f"{r['baseline_conf']:.4f}",
                                "Candidate Conf": f"{r['candidate_conf']:.4f}",
                                "Delta": f"{r['delta_conf']:+.4f}",
                                "Required": f">={r['required_conf']:.2f}",
                                "Note": r.get("note", ""),
                            })
                        st.dataframe(reg_table, width="stretch")

                    btn_col1, btn_col2, _ = st.columns([2, 2, 4])
                    with btn_col1:
                        if st.button(
                            "🚀 Approve & Promote to Production",
                            key=f"appr_model_{c_id}",
                            type="primary",
                            width="stretch",
                        ):
                            try:
                                res = approve_and_deploy_model(cand_folder)
                                st.session_state["deploy_success_msg"] = (
                                    f"🎉 **Model Successfully Promoted to Production!** Version: `{res['version']}`"
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deploying model: {e}")

                    with btn_col2:
                        if st.button(
                            "🗑️ Reject & Discard",
                            key=f"rej_model_{c_id}",
                            width="stretch",
                        ):
                            try:
                                reject_staged_model(cand_folder)
                                st.warning(f"Candidate `{c_id}` discarded.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error discarding candidate: {e}")

                    st.markdown("---")

        # 3. Model Version History Table
        st.markdown("#### 📜 Semantic Version Deployment History")
        history = active_info.get("history", [])
        if history:
            hist_rows = []
            for h in reversed(history):
                m = h.get("metrics", {})
                map50_val = m.get("mAP50")
                prec_val = m.get("precision")
                rec_val = m.get("recall")

                hist_rows.append(
                    {
                        "Version": str(h.get("version", "")),
                        "Model File": str(h.get("model_file", "")),
                        "Deployed At": str(h.get("deployed_at", "")),
                        "Training Samples": str(h.get("training_samples", "-")),
                        "mAP@0.5": f"{map50_val:.4f}" if isinstance(map50_val, (int, float)) else "-",
                        "Precision": f"{prec_val:.4f}" if isinstance(prec_val, (int, float)) else "-",
                        "Recall": f"{rec_val:.4f}" if isinstance(rec_val, (int, float)) else "-",
                        "Status": "ACTIVE" if h.get("version") == active_ver else "ARCHIVED",
                    }
                )
            st.dataframe(hist_rows, width="stretch")

            # 6. Model Reversion / Downgrade Section
            st.markdown("---")
            st.markdown("#### ⏪ Revert / Downgrade Active Production Model")
            st.caption(
                "If a newly promoted model degrades counting accuracy or causes field regressions, you can instantly rollback production to any historical version."
            )

            revertible_versions = [
                h for h in reversed(history) if h.get("version") and h.get("version") != active_ver
            ]

            if not revertible_versions:
                st.info("ℹ️ No previous versions available to revert to.")
            else:
                ver_options = [h["version"] for h in revertible_versions]
                rev_col1, rev_col2 = st.columns([3, 2])
                with rev_col1:
                    target_ver = st.selectbox(
                        "Select Target Version to Rollback to:",
                        options=ver_options,
                        format_func=lambda v: f"{v} (Artifact: {next((h.get('model_file') for h in revertible_versions if h.get('version') == v), '')})",
                        key=f"sel_revert_from_{active_ver}",
                    )

                target_entry = next((h for h in revertible_versions if h.get("version") == target_ver), None)
                if target_entry:
                    t_m = target_entry.get("metrics", {})
                    st.info(
                        f"**Target Version `{target_ver}` Specs:** "
                        f"File: `{target_entry.get('model_file')}` | "
                        f"Deployed: `{target_entry.get('deployed_at')}` | "
                        f"mAP@0.5: `{t_m.get('mAP50', '-')}` | "
                        f"Precision: `{t_m.get('precision', '-')}` | "
                        f"Recall: `{t_m.get('recall', '-')}`"
                    )

                with rev_col2:
                    st.write("")
                    st.write("")
                    revert_btn = st.button(
                        f"⏪ Rollback to {target_ver}",
                        type="secondary",
                        width="stretch",
                        key=f"btn_revert_{active_ver}_to_{target_ver}",
                    )

                if revert_btn:
                    try:
                        res = revert_to_version(target_ver)
                        st.session_state["revert_success_msg"] = (
                            f"🎉 **Production model successfully reverted to `{res['reverted_to']}`!** "
                            f"Active pointer `models/yolov8_traffic_latest.pt` now runs `{res['model_file']}`."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error reverting model: {e}")

            # 5. Golden Test Set Management Section
            st.markdown("---")
            st.markdown("#### ⭐ Golden Test Set Registry (Regression Suite)")
            st.caption(
                "Permanent benchmark test cases saved from difficult edge-cases (e.g. shadow transitions, lighting dips). "
                "Any newly staged model must undergo regression testing against these cases before promotion."
            )
            g_cases = load_golden_cases()
            if g_cases:
                g_rows = []
                for idx, c in enumerate(g_cases, 1):
                    g_rows.append({
                        "No.": idx,
                        "Case ID": c.get("case_id"),
                        "Target Class": c.get("target_class"),
                        "Baseline Version": c.get("baseline_version"),
                        "Baseline Conf": f"{c.get('baseline_confidence', 0.0):.4f}",
                        "Req Threshold": f">={c.get('min_required_conf', 0.35):.2f}",
                        "Note": c.get("note", ""),
                        "Added At": c.get("added_at", ""),
                    })
                st.dataframe(g_rows, width="stretch")

                # Case Detail Inspector
                st.markdown("##### 🔍 Inspect Golden Case")
                case_options = [c.get("case_id") for c in g_cases]
                selected_case_id = st.selectbox(
                    "Select Golden Test Case to inspect details:",
                    options=case_options,
                    format_func=lambda cid: f"{cid} ({next((c.get('target_class') for c in g_cases if c.get('case_id') == cid), '')})",
                    key="sel_golden_case_inspect",
                )

                selected_case = next((c for c in g_cases if c.get("case_id") == selected_case_id), None)
                if selected_case:
                    gc_col1, gc_col2 = st.columns([3, 2])
                    with gc_col1:
                        prev_rgb = draw_golden_case_preview(selected_case)
                        if prev_rgb is not None:
                            st.image(
                                prev_rgb,
                                caption=f"Golden Case: {selected_case_id} — Target: {selected_case.get('target_class')} (Red Box)",
                                width="stretch",
                            )
                        else:
                            st.warning("Case image preview not available.")
                    with gc_col2:
                        st.markdown(f"**Case ID:** `{selected_case.get('case_id')}`")
                        st.markdown(f"**Target Class:** `{selected_case.get('target_class')}`")
                        st.markdown(f"**Frame Index:** `#{selected_case.get('frame_idx', 'N/A')}`")
                        st.markdown(f"**Original Alert:** `{selected_case.get('reason', 'N/A')}`")
                        st.markdown(f"**Baseline Model:** `{selected_case.get('baseline_version')}`")
                        st.markdown(f"**Baseline Confidence:** `{selected_case.get('baseline_confidence'):.4f}`")
                        st.markdown(f"**Required Threshold:** `>={selected_case.get('min_required_conf', 0.35):.2f}`")
                        st.markdown(f"**Target Bounding Box:** `{selected_case.get('target_bbox')}`")
                        st.markdown(f"**Note:** *{selected_case.get('note', '')}*")
                        st.markdown(f"**Registered At:** `{selected_case.get('added_at')}`")
            else:
                st.info("ℹ️ No Golden Test Cases registered yet. Go to **Tab 2: Detail Inspector** and click **⭐ Save to Golden Test Set** on any tricky candidate frame!")


    # --- TAB 5: RUN HISTORY & SUMMARY ---
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
                        sem_ver = h.get("semantic_version")
                        if not sem_ver:
                            m_raw = h.get("model_version", "")
                            sem_ver = "v1.0.0" if "yolov8n" in m_raw else "v1.1.0"
                        art_name = h.get("model_artifact")
                        if not art_name:
                            art_name = Path(h.get("model_version", "")).name

                        table_rows.append(
                            {
                                "Run ID": str(h.get("run_id")),
                                "Source": str(h.get("source_name")),
                                "Model Version": str(sem_ver),
                                "Model Artifact": str(art_name),
                                "Start Time": str(h.get("start_time")),
                                "Duration (s)": str(h.get("duration_seconds")),
                                "Frames": str(h.get("total_frames_processed")),
                                "IN": str(cnt.get("vehicles_in", 0)),
                                "OUT": str(cnt.get("vehicles_out", 0)),
                                "Candidates": str(snaps.get("total_captured", 0)),
                            }
                        )
                    st.dataframe(table_rows, width="stretch")
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
