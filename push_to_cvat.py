import argparse
from pathlib import Path

import config
from cvat_integration import push_approved_to_cvat, export_approved_to_zip


def parse_arguments():
    parser = argparse.ArgumentParser(description="Push Approved Active Learning Candidates to CVAT")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(config.CANDIDATES_DIR),
        help="Path to candidate run directory (default: candidates directory)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Custom name for the CVAT annotation task",
    )
    parser.add_argument(
        "--no-pre-annotate",
        dest="pre_annotate",
        action="store_false",
        default=True,
        help="Disable automatic pre-annotation with YOLO model",
    )
    parser.add_argument(
        "--export-zip-only",
        type=str,
        default=None,
        help="Export approved images to a local zip file instead of pushing to CVAT",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_dir = Path(args.run_dir)

    print("=" * 65)
    print("🚀 CVAT ACTIVE LEARNING PUSHER (WITH PRE-ANNOTATION)")
    print("=" * 65)
    print(f"Target Directory  : {run_dir}")
    print(f"CVAT Server URL   : {config.CVAT_URL}")
    print(f"CVAT Username     : {config.CVAT_USERNAME or '(Not configured in .env)'}")
    print(f"Pre-annotation    : {'Enabled (Auto-draw with YOLO)' if args.pre_annotate else 'Disabled'}")

    if args.export_zip_only:
        out_zip = Path(args.export_zip_only)
        print(f"\n📦 Exporting approved images to zip: {out_zip} ...")
        try:
            count, _ = export_approved_to_zip(run_dir, out_zip)
            print(f"✅ Successfully exported {count} approved images to: {out_zip}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
        return

    print("\nConnecting to CVAT and pushing approved candidates...")
    try:
        result = push_approved_to_cvat(
            run_dir=run_dir,
            task_name=args.task_name,
            enable_pre_annotation=args.pre_annotate,
        )

        if result.get("success"):
            print("\n🎉 SUCCESS! All approved candidates have been pushed to CVAT.")
            print(f"   • Task ID             : #{result['task_id']}")
            print(f"   • Task Name           : {result['task_name']}")
            print(f"   • Images Count        : {result['count']}")
            print(f"   • Pre-annotated Boxes : {result.get('pre_annotated_boxes', 0)} boxes pre-drawn by AI")
            print(f"   • Task Web URL        : {result['task_url']}")
            print("\n👉 Open the URL above in your browser: obvious vehicles are ALREADY pre-drawn!")
        else:
            print(f"\n⚠️ Notice: {result.get('error')}")

    except Exception as e:
        print(f"\n❌ Error during CVAT push: {e}")
        print("\nPlease check:")
        print(" 1. Is CVAT running? (e.g. docker ps / http://localhost:8080)")
        print(" 2. Are CVAT_USERNAME and CVAT_PASSWORD set correctly in your .env file?")
        print(f" 3. Test opening CVAT in browser: {config.CVAT_URL}")


if __name__ == "__main__":
    main()
