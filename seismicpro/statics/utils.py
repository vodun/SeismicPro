def get_uphole_correction_method(survey, uphole_correction_method):
    if uphole_correction_method not in {"auto", "time", "depth", None}:
        raise ValueError

    if uphole_correction_method == "auto":
        if not survey.is_uphole:
            return None
        return "time" if "SourceUpholeTime" in survey.available_headers else "depth"

    if uphole_correction_method == "time" and "SourceUpholeTime" not in survey.available_headers:
        raise ValueError
    if uphole_correction_method == "depth" and "SourceDepth" not in survey.available_headers:
        raise ValueError
    return uphole_correction_method
