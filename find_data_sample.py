import json
# code = "/scratch2/f0072r1/img_dataset/sots_haze/outdoor/clear/0014.png"

with open("/scratch2/f0072r1/res_gemma/full_data.json", "r") as f:
    fullds = json.load(f)


def find_sots():
    match_code = "sots_haze/outdoor"
    to_save = []
    to_sort = []
    for item in fullds:
        if match_code in item["hq_path"]:
            print(item)
            to_save.append(item)
            to_sort.append(item["hq_path"].split('/')[-1])
    tmp = sorted(range(len(to_sort)), key=lambda k: to_sort[k])
    # print(len(to_save))
    to_save_1 = []
    for id in tmp:
        to_save_1.append(to_save[id])
    with open(f"full_haze_outdoor.json", "w") as f:
        json.dump(to_save_1, f)

if __name__ == "__main__":
    find_sots()
# for item in fullds:
#     if item["hq_path"] == code:
#         print(item)