if not os.path.exists(model_path_decoder) or not os.path.exists(model_path_encoder):
    download_command = f"omz_downloader " \
                       f"--name {model_name} " \
                       f"--precision {precision} " \
                       f"--output_dir {base_model_dir}"
    ! $download_command