def print_log(id, target_filename, output_dir, cam_rot_delta, cam_trans_delta, start_error, sample_error, coarse_error, end_error, optimization_steps, loss):
    print("id:", id)
    print("target file name:", target_filename)
    print("delta_rot:", cam_rot_delta)
    print("delta_trans:", cam_trans_delta)

    print("output path: \033[34m{}\033[0m".format(output_dir))
    print("optimization steps:", optimization_steps)
    print("comparing loss:", loss)

    color_reset = "\033[0m"
    color_green = "\033[032m"
    color_red = "\033[031m"

    start_error_R = start_error["R"]
    start_error_t = start_error["t"]
    color_R = color_green if start_error_R < 5 else color_red
    color_t = color_green if start_error_t < 0.05 else color_red
    print("  start error - {}rotation: {:5.2f} degree.{} {}translation: {:5.2f} cm.{}".format(
        color_R, start_error_R, color_reset, color_t, start_error_t, color_reset))

    sample_error_R = sample_error["R"]
    sample_error_t = sample_error["t"]
    color_R = color_green if sample_error_R < 5 else color_red
    color_t = color_green if sample_error_t < 0.05 else color_red
    print(" sample error - {}rotation: {:5.2f} degree.{} {}translation: {:5.2f} cm.{}".format(
        color_R, sample_error_R, color_reset, color_t, sample_error_t, color_reset))
    
    coarse_error_R = coarse_error["R"]
    coarse_error_t = coarse_error["t"]
    color_R = color_green if coarse_error_R < 5 else color_red
    color_t = color_green if coarse_error_t < 0.05 else color_red
    print(" coarse error - {}rotation: {:5.2f} degree.{} {}translation: {:5.2f} cm.{}".format(
        color_R, coarse_error_R, color_reset, color_t, coarse_error_t, color_reset))

    end_error_R = end_error["R"]
    end_error_t = end_error["t"]
    color_R = color_green if end_error_R < 5 else color_red
    color_t = color_green if end_error_t < 0.05 else color_red
    print("  fine  error - {}rotation: {:5.2f} degree.{} {}translation: {:5.2f} cm.{}".format(
        color_R, end_error_R, color_reset, color_t, end_error_t, color_reset))
    
    print("*************************************")