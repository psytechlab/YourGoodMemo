def format_dialog(hist, step=None):
    if step is None:
        step = len(hist)
    dialog = [f"Друг: {ter}\n\nПользователь: {clt}" for ter, clt in hist[: step]]
    return "\n\n".join(dialog)

