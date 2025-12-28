from scripts.train import entry_point


if __name__ == "__main__":
    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"Running with p_empty = {i}")
        entry_point(p_empty=i, p_rand=0.0)
    for j in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"Running with p_rand = {j}")
        entry_point(p_empty=0.0, p_rand=j)
