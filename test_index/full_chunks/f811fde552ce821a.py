def process_folder(folder_data):
        """Process a single folder summary."""
        folder, files = folder_data
        folder_sum = summarize_folder(files)
        folder_chunk = {
            "text": f"FOLDER: {folder}\n{folder_sum}",
            "metadata": {
                "folder": folder,
                "level": "folder_summary",
                "location": {"folder": folder},
            },
        }
        return folder_chunk