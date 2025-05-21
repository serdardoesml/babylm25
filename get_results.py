# get_results.py
import os
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--backend", type=str, default="mlm", choices=["mlm", "causal", "mntp"])
parser.add_argument("--revision", type=str, default="main")
parser.add_argument("--task", type=str, default="all")
parser.add_argument("--fast", action="store_true", help="Fast results")
parser.add_argument("--no_key", action="store_true", help="Dont print key")

def get_avg_accuracy(file):
    # get last non-empty line
    with open(file, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if line.strip() != "":
                return float(line)

def get_reading_accuracy(file):
    results = {}
    with open(file, "r") as f:
        results["eye_tracking"] = float(f.readline().split(":")[1])          
        results["self-paced_reading"] = float(f.readline().split(":")[1])
    return results

def get_glue_results(file):
    results = {}
    with open(file, "r") as f:
        results["accuracy"] = float(f.readline().split(":")[1])
        try: # check if there is a second line, mnli doesnt report f1
            results["f1"] = float(f.readline().split(":")[1])
        except IndexError:
            pass
        # results["mcc"] = float(f.readline().split(":")[1])
    return results

def zero_shot(args):
    results = {}
    zs_path = os.path.join(args.path, args.revision, "zero_shot", args.backend)
    
    reports = list(pathlib.Path(zs_path).rglob("report.txt")) + list(pathlib.Path(zs_path).rglob("best_temperature_report.txt"))
    print(reports)
    for report in reports:    
        parent_folder = pathlib.Path(report).parent.stem
        if "_fast" in parent_folder and not args.fast:
            continue
        if "_fast" not in parent_folder and args.fast:
            continue
        if args.task != "all" and args.task != parent_folder:
            continue

        if "best_temperature_report" in str(report):
            accuracy = get_avg_accuracy(report)
            results[parent_folder] = accuracy
        else:
            results.update(get_reading_accuracy(report))

    # sort results in order: blimp, blimp_supp, ewok, self-paced, entity, wugs
    order = ["blimp_filtered", "supplement_filtered", "ewok", "eye_tracking", "self-paced_reading", "entity_tracking", "wug_adj_nominalization"]
    
    print(list(results.keys()))
    # print results in a nice table
    print("Dataset\tAccuracy")
    for key in order:
        if key in results:
            if args.no_key:
                print(f"{results[key]:.2f}")
            else:
                print(f"{key}\t{results[key]:.2f}")

def finetune(args):
    results = {}
    ft_path = os.path.join(args.path, args.revision, "finetune")

    reports = list(pathlib.Path(ft_path).rglob("results.txt"))
    for report in reports:
        parent_folder = pathlib.Path(report).parent.stem
        if args.task != "all" and args.task != parent_folder:
            continue

        print(report)
        if parent_folder in ["mrpc", "qqp"]:
            results[parent_folder] = get_glue_results(report)['f1']
        else:
            results[parent_folder] = get_glue_results(report)['accuracy']


    # sort results in order: blimp, blimp_supp, ewok, self-paced, entity, wugs
    order = ["boolq", "mnli", "mrpc", "qqp", "multirc", "rte", "wsc"]
    
    print(list(results.keys()))
    # print results in a nice table
    print("Dataset\tScore")
    for key in order:
        if key in results:
            if args.no_key:
                print(f"{results[key]*100:.2f}")
            else:
                print(f"{key}\t{results[key]*100:.2f}")


def main():
    args = parser.parse_args()
    if args.task == "all" or args.task == "zs":
        zero_shot(args)
    elif args.task == "all" or args.task == "ft":
        finetune(args)

if __name__ == "__main__":
    main()