
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_experiment_results(filepath):
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    experiments = []
    current_exp = None
    corrections = 0
    correct_actions = 0
    total_experiments = 0
    corrections_per_object = defaultdict(int)  # total number of corrections (all sentences)
    corrections_sentences_per_object = defaultdict(list)
    corrections_per_experiment = []
    corrections_sentences_per_experiment = []
    object_correction_experiment_count = defaultdict(int)  # number of experiments with at least one correction for this object

    current_object = None
    current_exp_correction_sentences = 0
    objects_with_correction_in_exp = set()
    for line in lines:
        if line.startswith("===== New Experiment"):
            # At the end of each experiment, update object_correction_experiment_count
            for obj in objects_with_correction_in_exp:
                object_correction_experiment_count[obj] += 1
            objects_with_correction_in_exp = set()
            if current_exp is not None:
                experiments.append(current_exp)
                corrections_per_experiment.append(current_exp["corrections"])
                corrections_sentences_per_experiment.append(current_exp_correction_sentences)
            current_exp = {
                "corrections": 0,
                "rl_result": "",
            }
            total_experiments += 1
            current_object = None
            current_exp_correction_sentences = 0
        elif line.strip().startswith("Object: "):
            # Track the current object
            match_obj = re.match(r'Object: ([a-zA-Z ]+)', line.strip())
            if match_obj:
                current_object = match_obj.group(1).strip()
        elif "Corrections needed" in line:
            match = re.search(r'Corrections needed \((\d+)\)', line)
            if match and current_object:
                n_corr = int(match.group(1))
                current_exp["corrections"] += n_corr
                corrections += n_corr
                corrections_per_object[current_object] += n_corr
                corrections_sentences_per_object[current_object].append(n_corr)
                current_exp_correction_sentences += n_corr
                objects_with_correction_in_exp.add(current_object)
        elif line.startswith("RL result") or line.startswith("RL results"):
            current_exp["rl_result"] = line.strip()
            if "all actions are correct" in line:
                correct_actions += 1

    # Add last experiment
    if current_exp is not None:
        experiments.append(current_exp)
        corrections_per_experiment.append(current_exp["corrections"])
        corrections_sentences_per_experiment.append(current_exp_correction_sentences)
        for obj in objects_with_correction_in_exp:
            object_correction_experiment_count[obj] += 1

    print(f"Total number of experiments: {total_experiments}")
    print(f"Number of experiments without correction: {sum(1 for e in experiments if e['corrections'] == 0)}")
    print(f"Total number of corrections: {corrections}")
    print(f"Number of experiments with 'all actions are correct': {correct_actions}")
    print(f"Performance (experiments without correction / total): {sum(1 for e in experiments if e['corrections'] == 0) / total_experiments:.2%}")

    print("\nTotal correction sentences per object:")
    for obj, num in corrections_per_object.items():
        avg = sum(corrections_sentences_per_object[obj]) / len(corrections_sentences_per_object[obj]) if corrections_sentences_per_object[obj] else 0
        print(f"{obj}: {num} (avg per experiment: {avg:.2f})")

    print("\nNumber of experiments with at least one correction for each object:")
    for obj, count in object_correction_experiment_count.items():
        print(f"{obj}: {count}")

    # Plot 1: Bar chart of total corrections per object
    if corrections_per_object:
        plt.figure(figsize=(8, 5))
        plt.bar(corrections_per_object.keys(), corrections_per_object.values(), color='skyblue')
        plt.xlabel('Object')
        plt.ylabel('Number of correction sentences (all experiments)')
        plt.title('Total correction sentences per object')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig('results/plot_total_corrections_per_object.png')
        plt.show()


    # Plot 2: Histogram of number of correction sentences per experiment
    if corrections_sentences_per_experiment:
        plt.figure(figsize=(7, 4))
        plt.hist(corrections_sentences_per_experiment, bins=range(0, max(corrections_sentences_per_experiment)+2), color='orange', edgecolor='black', align='left')
        plt.xlabel('Number of correction sentences per experiment')
        plt.ylabel('Number of experiments')
        plt.title('Distribution of correction sentences per experiment')
        plt.tight_layout()
        plt.savefig('results/plot_hist_corrections_per_experiment.png')
        plt.show()

    # Plot 3: Violin plot of correction sentences per object (better for small N)
    if corrections_sentences_per_object:
        plt.figure(figsize=(8, 5))
        data = [v for v in corrections_sentences_per_object.values()]
        plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(corrections_sentences_per_object) + 1), list(corrections_sentences_per_object.keys()), rotation=30)
        plt.xlabel('Object')
        plt.ylabel('Correction sentences per experiment')
        plt.title('Correction sentences per object (violin plot)')
        plt.tight_layout()
        plt.savefig('results/plot_violin_corrections_per_object.png')
        plt.show()

    # Plot 4: Line plot of cumulative precision as a function of number of corrections, per object and combined
    if corrections_sentences_per_object:
        from collections import Counter
        plt.figure(figsize=(8, 5))
        max_corr_all = max([max(lst) for lst in corrections_sentences_per_object.values() if lst]+[0])
        # Per-object curves
        for obj, corrections_list in corrections_sentences_per_object.items():
            if not corrections_list:
                continue
            max_corr = max(corrections_list)
            count_hist = Counter(corrections_list)
            total = len(corrections_list)
            cumulative = []
            cum_sum = 0
            for n in range(0, max_corr+1):
                cum_sum += count_hist.get(n, 0)
                cumulative.append(cum_sum / total)
            plt.plot(range(0, max_corr+1), cumulative, marker='o', label=obj)
        # Combined curve (all object-correction pairs)
        all_corrections = [v for lst in corrections_sentences_per_object.values() for v in lst]
        if all_corrections:
            count_hist_all = Counter(all_corrections)
            total_all = len(all_corrections)
            cumulative_all = []
            cum_sum_all = 0
            for n in range(0, max_corr_all+1):
                cum_sum_all += count_hist_all.get(n, 0)
                cumulative_all.append(cum_sum_all / total_all)
            plt.plot(range(0, max_corr_all+1), cumulative_all, marker='o', color='black', linestyle='--', label='All objects combined')
        plt.xlabel('Maximum number of corrections allowed (per object)')
        plt.ylabel('Cumulative precision')
        plt.title('Cumulative precision vs. number of corrections per object')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(range(0, max_corr_all+1))
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/plot_cumulative_precision.png')
        plt.show()

    # Print more info about correction sentences
    print("\nCorrection sentences per experiment (first 10):", corrections_sentences_per_experiment[:10])
    print("Average correction sentences per experiment:", sum(corrections_sentences_per_experiment)/len(corrections_sentences_per_experiment) if corrections_sentences_per_experiment else 0)

if __name__ == "__main__":
    parse_experiment_results("/home/ines/RAG/llm_vlm_planner/results/experiment_results.txt")