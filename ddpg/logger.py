import time


def log_eval(
    eval_idx: int,
    num_evals: int,
    environment_steps: int,
    training_steps: int,
    average_reward: float,
    train_start_time: float,
    line_width: int = 80,
):
    """Print progress after one training/evaluation cycle."""

    elapsed_seconds = int(time.time() - train_start_time)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_seconds, 60)
    elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
    elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"

    header = f"Eval {eval_idx}/{num_evals}"
    print("")
    print("#" * line_width)
    print(f"\033[1m{header.center(line_width)}\033[0m")
    print(
        f"{f'Environment steps: {environment_steps}/{training_steps}'.center(line_width)}"
    )
    print(f"{f'Mean reward: {average_reward:.2f}'.center(line_width)}")
    print(f"{f'Time elapsed: {elapsed_str}'.center(line_width)}")
    print("-" * line_width)
