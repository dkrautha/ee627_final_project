import csv
import glob
import os

SUBMISSIONS_FOLDER = "submissions"

data_list = {
    os.path.splitext(os.path.basename(file))[0].split("-")[1]: dict(
        csv.reader(open(file))
    )
    for file in glob.glob("submissions/results-*.csv")
}
for i, fileone_ver in enumerate(data_list):
    is_duplicate = False
    for j, filetwo_ver in enumerate(list(data_list.keys())[i + 1 :]):
        # if i == j:
        #     continue
        is_identical = True  # assume identical, look for counterexamples
        for key, val in data_list[fileone_ver].items():
            if data_list[filetwo_ver][key] != val:
                is_identical = False
                break
        if is_identical:
            print(f"{fileone_ver} is the same as {filetwo_ver}")
            is_duplicate = True
            break
print("Done")
