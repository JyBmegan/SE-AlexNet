import os
import time

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error: {cmd}")
        return False
    else:
        print("Done!")
        return True

def main():
    se_pos = [1, 2, 3, 4]
    ratios = [32, 16, 8, 4, 2]
    trans_con = ['Face', 'Object']
    masks = ['Full', 'E', 'N', 'M']

    completed_task = [
        ('Face', 1, 8),
        ('Face', 1, 16),
        ('Face', 1, 32)
    ]

    total_tasks = len(trans_con) * len(se_pos) * len(ratios)
    current_task_idx = 0

    for trans in trans_con:
        for pos in se_pos:
            for r in ratios:
                current_task_idx += 1

                if (trans, pos, r) in completed_task:
                    print(f"[{current_task_idx}/{total_tasks}] Skip this FINISHED task: {trans} | Pos {pos} | Ratio {r}")
                    continue

                print(f"[{current_task_idx}/{total_tasks}] New task: {trans} | Pos {pos} | Ratio {r}")

                # Train
                cmd_train = f"python train.py --se_pos {pos} --reduction {r} --trans {trans}"
                success = run_command(cmd_train)
                if not success:
                    print ("Error! Train Next Con")
                    continue

                # Predict *4 mask
                print(f"Now predict with 4 mask faces")
                for mask in masks: 
                    cmd_pred = f"python predict.py --se_pos {pos} --reduction {r} --trans {trans} --mask {mask}"
                    run_command(cmd_pred)

                # GradCAM *4 mask
                print(f"Now run GradCAM with 4 mask faces")
                for mask in masks:
                    cmd_cam = f"python gradcam.py --se_pos {pos} --reduction {r} --trans {trans} --mask {mask}"
                    run_command(cmd_cam)

                time.sleep(10)

if __name__ == '__main__':
    main()

