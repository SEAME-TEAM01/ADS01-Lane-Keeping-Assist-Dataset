import  os
import  cv2
import  json
import  numpy as np

def draw_lanes_on_image(dataset):
    lanes_x     = dataset["lanes_x"]
    lane_y      = dataset["lane_y"]
    mask        = dataset["mask"]
    mask        = mask.replace("../../dataset/datasetv2/trainset/masks", "masks_new")

    mask_new    = np.zeros((720, 1280, 3), dtype=np.uint8)

    lanes = []
    for i in range(0, len(lanes_x)):
        xs = []
        ys = []
        for j in range(0, len(lane_y)):
            x = lanes_x[i][j]
            y = lane_y[j]
            if x == -2:
                continue
            xs.append(x)
            ys.append(y)
        if xs and ys:
            lanes.append((xs, ys))

    for lane in lanes:
        xs = lane[0]
        ys = lane[1]
        for i in range(0, len(xs)):
            x = xs[i]
            y = ys[i]
            if i > 0:
                prev_x, prev_y = xs[i-1], ys[i-1]
                cv2.line(mask_new, (prev_x, prev_y), (x, y), color=(255, 255, 255), thickness=15)
    
    folder = os.path.dirname(mask)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    cv2.imwrite(mask, mask_new)
    print(f"[INFO] Image saved: {mask}")

if __name__ == "__main__":
    datasets = open("labels/train_gt_tmp.json")
    for dataset in datasets:
        dataset = json.loads(dataset)
        draw_lanes_on_image(dataset)