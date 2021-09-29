#### JS to download all the urls from the page: https://www.spec.org/cpu2017/results/cpu2017.html
#### Save the results in the corresponding text files and remove the '[' and ']' chars.
# all_hrefs = []
# all_csvs = document.getElementById("CINT2017_speeddiv").querySelectorAll('table > tbody> tr > td > span > a[href*="csv"]');console.log(all_csvs.length)
# for(var i=0; i<all_csvs.length;i++) {all_hrefs[i] = all_csvs[i].href}
# console.log(JSON.stringify(all_hrefs))
####

#### Python to download all the files from a list in a txt file
# %reset -s -f
# import os, time
# import urllib.request

# category = 'choose!!'
# #category = 'FP_rate'
# #category = 'FP_speed'
# #category = 'Int_rate'
# #category = 'Int_speed'

# text_file = open(category + ".txt", "r")
# lines = text_file.read().replace('"', '').split(',')
# print(len(lines))

# # Download the file from `url` and save it locally under `dl_file`:
# for i in range (0, len(lines)):
#     dl_file = category+"_CSVs/" + lines[i].rsplit('/', 1)[1]
#     if not os.path.isfile(dl_file):
#         urllib.request.urlretrieve(lines[i], dl_file)
#         time.sleep(0.02)
# print("done")

####
import os
import csv
import numpy as np
import pandas as pd
import re  # regex

# from IPython.display import display # for Jupyter
# pd.set_option('display.max_columns', None) # show all columns when display
# pd.set_option('display.max_rows', None)


def write_cvs_to_inputs(suite, data_rec, RECOM=False, RATING=None):
    """
    Loads from SPEC 2017 results CSV files and populates the input files. 
    RECOM and RATING parameters are only valid for the Recommendation part
    """
    # for recom
    if RECOM == True:
        min_Rate, min_Time, max_Rate, max_Time = {}, {}, {}, {}

    cat = suite["name"]
    for bench in suite["benchmarks"]:
        data_rec[bench] = ""
        data_rec["t_" + bench] = ""

        if RECOM == False:
            with open("data/in/input_" + cat + ".csv", "w") as csvfile:
                csvfile.write(",".join(data_rec.keys()) + "\n")
        # for recom
        else:
            min_Rate[bench] = 10 ** 30
            max_Rate[bench] = 0
            min_Time["t_" + bench] = 10 ** 30
            max_Time["t_" + bench] = 0
            with open("data/in/recom_input_" + cat + ".csv", "w") as csvfile:
                csvfile.write("user,item,Rate,Time\n")  # NO space

    CSVs_path = "data/" + cat + "_CSVs/"
    for f in os.listdir(CSVs_path):
        # automatically ignores commas inside "str,str".
        # also 'encoding' and 'errors' replace strange characters down in Notes
        with open(CSVs_path + f, encoding="utf-8", errors="replace") as csv_inp:
            reader = csv.reader(csv_inp)

            # Clear values of of the dict
            data_rec = {key: "" for key in data_rec}

            # benchmark_start = 0
            # benchmark_end = 0
            sel_bench_start = False
            sel_bench_end = False

            try:
                # Tip: make a list of rows to use indexes from csv.reader
                reader_list = list(reader)
                # Iterate over each row in the CSV
                # if you need the index to mark benchmark_start and end, then
                # `for i,row in enumerate(reader):``
                for ind, row in enumerate(reader_list):
                    if len(row) > 0:
                        if row[0] == "Selected Results Table":
                            sel_bench_start = True

                        if cat == "CINT2017_speed":
                            if row[0] == "SPECspeed2017_int_base":
                                sel_bench_end = True

                        elif cat == "CFP2017_speed":
                            if row[0] == "SPECspeed2017_fp_base":
                                sel_bench_end = True

                        elif cat == "CINT2017_rate":
                            if row[0] == "SPECrate2017_int_base":
                                sel_bench_end = True

                        elif cat == "CFP2017_rate":
                            if row[0] == "SPECrate2017_fp_base":
                                sel_bench_end = True

                        # Selected Benchmark Results area
                        if sel_bench_start == True and sel_bench_end == False:
                            # NOTE: It seems that all benchmarks on the same machine
                            # are run with the same number of threads
                            # Do not read empty values, let Pandas put NaN.
                            # Otherwise you will add empty strings to the dataset.
                            if row[0] == suite["benchmarks"][0] and row[1] != " ":
                                if row[1] == "NC":
                                    break  # break if Not Compliant (NC)
                                try:
                                    data_rec["threads_or_copies"] = int(row[1])
                                except:
                                    # a case like '16/2', just mark it as NC by breaking
                                    break

                            # row[2] = Base Run Time, row[3] = Base Ratio
                            for bench in suite["benchmarks"]:
                                if row[0] == bench and row[3] != " ":
                                    data_rec["t_" + bench] = float(row[2])
                                    data_rec[bench] = float(row[3])

                        # Hardware vendor is different from vendor
                        if row[0] == "Hardware Vendor:" and row[1] != " ":
                            data_rec["hardware_vendor"] = row[1].strip()
                        # Grab the vendor and machine model name
                        if row[0] == "CPU Name" and row[1] != " ":
                            full_name = row[1].split(" ")
                            data_rec["vendor"] = full_name[0].strip()
                            data_rec["model_name"] = " ".join(full_name[1:])

                        if row[0] == "Test sponsor:" and row[1] != " ":
                            data_rec["test_sponsor"] = row[1].strip()

                        # Do not read empty values, let Pandas put NaN.
                        # Otherwise you will add empty strings to the dataset.
                        # You know you have done it right if the type of
                        # max_mhz in the dataframe becomes 'float64' and not 'object'.
                        # If it is object, you are probably adding a string somewhere
                        if row[0].strip().startswith("Max MHz") and row[1] != " ":
                            data_rec["max_mhz"] = row[1]
                        if row[0] == "  Nominal" and row[1] != " ":
                            data_rec["nominal_mhz"] = row[1]

                        # form lscpu:
                        if "Architecture:" in row[0]:
                            data_rec["arch"] = (
                                row[0].strip().split("Architecture:", 1)[1].strip()
                            )
                        if (
                            "CPU(s):" in row[0]
                            # only the row which contains CPU(s): xx,
                            # so the first bit after the split should be empty
                            and row[0].strip().split("CPU(s):", 1)[0] == ""
                        ):
                            data_rec["cpus"] = int(
                                row[0].strip().split("CPU(s):", 1)[1].strip()
                            )
                        if "Thread(s) per core:" in row[0]:
                            data_rec["threads_per_core"] = int(
                                row[0]
                                .strip()
                                .split("Thread(s) per core:", 1)[1]
                                .strip()
                            )
                        if "Core(s) per socket:" in row[0]:
                            data_rec["cores_per_socket"] = int(
                                row[0]
                                .strip()
                                .split("Core(s) per socket:", 1)[1]
                                .strip()
                            )
                        if "Socket(s):" in row[0]:
                            data_rec["sockets"] = int(
                                row[0].strip().split("Socket(s):", 1)[1].strip()
                            )
                        if "NUMA node(s):" in row[0]:
                            data_rec["numas"] = int(
                                row[0].strip().split("NUMA node(s):", 1)[1].strip()
                            )

                        if (
                            "NAME=" in row[0]
                            and row[0].strip().split("NAME=", 1)[0].strip() == ""
                        ):  # only the row which contains the exact 'NAME=xxx'
                            os_name = (
                                row[0]
                                .strip()
                                .split("NAME=", 1)[1]
                                .replace('"', "")
                                .strip()
                            )
                            if os_name.startswith("SLE"):
                                # TODO: IMPORTANT: IN order to get rid of SLES_HPC
                                # outliers, we needed to make this change
                                os_name = "SLES"
                            if os_name.startswith("Red Hat Enterprise Linux"):
                                os_name = "RHEL"
                                # TODO: IMPORTANT: all the OSs in all categories are
                                # RHEL Servers, we chose this name, because there are
                                # a few in the CINT_SPEED that are not servers
                                # and they ruin the game!
                            if os_name.startswith("CentOS"):
                                os_name = "CentOS"
                            data_rec["os_name"] = os_name
                        if (
                            "VERSION_ID=" in row[0]
                            and row[0].strip().split("VERSION_ID=", 1)[0].strip() == ""
                        ):
                            os_vid = (
                                row[0]
                                .strip()
                                .split("VERSION_ID=", 1)[1]
                                .replace('"', "")
                                .strip()
                            )
                            data_rec["os_vid"] = os_vid

                        if row[0] == "Cache L1" and row[1] != " ":
                            info = row[1].split(" ")
                            if info[1] == "KB" and info[2] == "I":
                                data_rec["l1i_cache_kb"] = info[0]
                            elif info[1] == "MB" and info[2] == "I":
                                data_rec["l1i_cache_kb"] = float(info[0]) * 1024
                            if info[5] == "KB" and info[6] == "D":
                                data_rec["l1d_cache_kb"] = info[4]
                            elif info[5] == "MB" and info[6] == "D":
                                data_rec["l1d_cache_kb"] = float(info[4]) * 1024

                        if row[0].strip() == "L2" and row[1] != " ":
                            info = row[1].strip().split(" ")
                            if info[1] == "KB" and info[2] == "I+D":
                                data_rec["l2_cache_kb"] = info[0]
                            elif info[1] == "MB" and info[2] == "I+D":
                                data_rec["l2_cache_kb"] = float(info[0]) * 1024

                        if row[0].strip() == "L3" and row[1] != " ":
                            info = (
                                row[1].strip().split(" ")
                            )  # seen cases like ' xx MB', hence strip()
                            if info[1] == "KB" and info[2] == "I+D":
                                data_rec["l3_cache_kb"] = info[0]
                            elif info[1] == "MB" and info[2] == "I+D":
                                data_rec["l3_cache_kb"] = float(info[0]) * 1024
                            elif info[1] == "GB" and info[2] == "I+D":
                                data_rec["l3_cache_kb"] = float(info[0]) * 1024 * 1024

                        # Note: since memory info is sometimes incorrect,
                        # we first check to see if meminfo is available
                        if "MemTotal:" in row[0]:
                            # 2 splits as it is in the format: xxx kB
                            data_rec["mem_kb"] = int(
                                row[0].strip().split("MemTotal:")[1].split()[0].strip()
                            )
                        if row[0] == "Memory" and row[1] != " ":
                            info = row[1].strip().split(" ")
                            if (
                                data_rec["mem_kb"] == ""
                            ):  # if not filled by info from meminfo
                                if info[1] == "GB":
                                    data_rec["mem_kb"] = float(info[0]) * 1024 * 1024
                                elif info[1] == "TB":
                                    data_rec["mem_kb"] = (
                                        float(info[0]) * 1024 * 1024 * 1024
                                    )
                                else:
                                    # there is an instance like that in CFP_speed
                                    # where they put MB instead of GB
                                    data_rec[
                                        "threads_or_copies"
                                    ] = ""  # Treat the case like an 'NC' case and break
                                    # raise ValueError('Memory not in GB, File: ', f)
                                    break
                            # TODO: Double check the memory channels with someone!
                            if len(info) > 2 and info[2] != " ":
                                data_rec["mem_channels"] = info[2][1:]
                            if len(info) > 5 and info[4] != " ":
                                if info[5] == "MB":
                                    data_rec["channel_kb"] = float(info[4]) * 1024
                                elif info[5] == "GB":
                                    data_rec["channel_kb"] = (
                                        float(info[4]) * 1024 * 1024
                                    )
                                elif info[5] == "TB":
                                    data_rec["channel_kb"] = (
                                        float(info[4]) * 1024 * 1024 * 1024
                                    )

                            for s in info:
                                if "-" in s:
                                    m = re.search("-(.+?)-", s)
                                    if (
                                        m
                                    ):  # like above, we deal with empty records later
                                        found = m.group()
                                        try:
                                            # it will raise an exception if not possible
                                            speed = int(
                                                re.search(r"\d+", found).group()
                                            )
                                            data_rec["mem_data_rate"] = speed
                                        except:
                                            raise ValueError(
                                                "Mem speed not int, File: ", f
                                            )

                        if row[0] == "Storage" and row[1] != " ":
                            info = row[1].split(" ")
                            for i in range(len(info)):
                                try:
                                    if (
                                        info[i].lower().replace(",", "") == "gb"
                                    ):  # seen cases like Gb!
                                        data_rec["storage_gb"] = float(info[i - 1])
                                        if "storage_type" in data_rec:
                                            data_rec["storage_type"] = " ".join(
                                                info[i + 1 :]
                                            )
                                        if (
                                            " ".join(info[i + 1 :])
                                            == "ZFS mirror on 2x 15K RPM 300 GB"
                                        ):  # Baeline Sun machine
                                            # pc100 = 100MT/s
                                            # https://en.wikipedia.org/wiki/CAS_latency
                                            data_rec["mem_data_rate"] = 100
                                        # to avoid cases where there are two GBs,
                                        # one at the end of the info with nothing after!
                                        break
                                    elif info[i].lower().replace(",", "") == "tb":
                                        data_rec["storage_gb"] = (
                                            float(info[i - 1]) * 1024
                                        )
                                        if "storage_type" in data_rec:
                                            data_rec["storage_type"] = " ".join(
                                                info[i + 1 :]
                                            )
                                    elif (
                                        "gb" in info[i].lower()
                                    ):  # or if information is like xxGB
                                        data_rec["storage_gb"] = float(
                                            info[i].replace(",", "")[:-2]
                                        )
                                        if "storage_type" in data_rec:
                                            data_rec["storage_type"] = " ".join(
                                                info[i + 1 :]
                                            )
                                    elif "tb" in info[i].lower():
                                        data_rec["storage_gb"] = (
                                            float(info[i].replace(",", "")[:-2]) * 1024
                                        )
                                        if "storage_type" in data_rec:
                                            data_rec["storage_type"] = " ".join(
                                                info[i + 1 :]
                                            )
                                except:
                                    # Treat the case like an 'NC' case and
                                    # break (seen invalid entries like: 1.92 TB GB)
                                    data_rec["threads_or_copies"] = ""
                                    # or raise ValueError('Storage problem, File: ', f)
                                    break

                        if row[0] == "OS" and row[1] != " ":
                            data_rec["os"] = row[1].strip()

                        if row[0] == "Compiler" and row[1] != " ":
                            if not "Intel" in row[1] and not "AOCC" in row[1]:
                                # Treat the case like an 'NC' case,
                                # NOTE: for now we just ignore a few complicated cases
                                data_rec["threads_or_copies"] = ""
                                break
                            # semicolon shifts the data when writing to csv
                            # using DictWriter (acts as a delimiter)
                            data_rec["compiler"] = row[1].strip().replace(";", "")

                        # Parallel: This field is automatically set to "Yes"
                        # if compiler flags are used that are marked with the
                        # parallel attribute, indicating that they cause
                        # either automatic or explicit parallelism.
                        if row[0] == "Parallel" and row[1] != " ":
                            data_rec["parallel"] = row[1]

                        if row[0] == "File System" and row[1] != " ":
                            if "ramfs" in row[1]:
                                # Treat the case like an 'NC' case,
                                # NOTE: for now we just ignore a few cases or ramfs
                                data_rec["threads_or_copies"] = ""
                                break
                            data_rec["file_system"] = row[1]

            except:
                raise ValueError("File has a problem: ", f)

            if data_rec["threads_or_copies"]:  # if not empty as a result of 'NC'
                data_rec[
                    "file"
                ] = f.rstrip()  # some files have a space after the extension!
                if RECOM == False:
                    with open("data/in/input_" + cat + ".csv", "a") as csvInp:
                        w = csv.DictWriter(csvInp, data_rec.keys())
                        # we don't need that as we have written the headers once
                        # w.writeheader()
                        w.writerow(data_rec)
                #####
                # recom
                else:  # write to recom
                    with open("data/in/recom_input_" + cat + ".csv", "a") as recInp:
                        for bench in data_rec:
                            if (
                                bench in suite["benchmarks"]
                            ):  # benchmarks are like  "users" in recom context
                                if RATING == "log":
                                    if np.log(data_rec[bench]) < min_Rate[bench]:
                                        min_Rate[bench] = np.log(data_rec[bench])
                                    if np.log(data_rec[bench]) > max_Rate[bench]:
                                        max_Rate[bench] = np.log(data_rec[bench])

                                    if (
                                        np.log(data_rec["t_" + bench])
                                        < min_Time["t_" + bench]
                                    ):
                                        min_Time["t_" + bench] = np.log(
                                            data_rec["t_" + bench]
                                        )
                                    if (
                                        np.log(data_rec["t_" + bench])
                                        > max_Time["t_" + bench]
                                    ):
                                        max_Time["t_" + bench] = np.log(
                                            data_rec["t_" + bench]
                                        )
                                else:
                                    if data_rec[bench] < min_Rate[bench]:
                                        min_Rate[bench] = data_rec[bench]
                                    if data_rec[bench] > max_Rate[bench]:
                                        max_Rate[bench] = data_rec[bench]

                                    if data_rec["t_" + bench] < min_Time["t_" + bench]:
                                        min_Time["t_" + bench] = data_rec["t_" + bench]
                                    if data_rec["t_" + bench] > max_Time["t_" + bench]:
                                        max_Time["t_" + bench] = data_rec["t_" + bench]

                                recInp.write(bench)
                                # we need unique systems ("items" in recom context) and
                                # the only unique field is 'file'!
                                recInp.write("," + data_rec["file"])
                                if RATING == "log":
                                    recInp.write("," + str(np.log(data_rec[bench])))
                                    recInp.write(
                                        "," + str(np.log(data_rec["t_" + bench]))
                                    )
                                else:
                                    recInp.write("," + str(data_rec[bench]))
                                    recInp.write("," + str(data_rec["t_" + bench]))
                                recInp.write("\n")
    # only for recom, return values too, which are the min and max Rates
    # (or rating range in a recommendation context)
    if RECOM == True:
        return min_Rate, max_Rate, min_Time, max_Time


def remove_nan(df, cols):
    # print(df[pd.isna(df.model_name)]) #TODO: handle that?
    for col in cols:
        df = df[~pd.isna(df[col])]
    #     df = df[df['threads_or_copies'] < 200]
    #     print("size after threads: ", len(df))
    return df


def find_anomalies(series):
    anomalies = []
    # Set upper and lower limit to 3 standard deviation
    series_std = np.std(series)
    series_mean = np.mean(series)
    anomaly_cut_off = series_std * 3

    lower_limit = series_mean - anomaly_cut_off
    upper_limit = series_mean + anomaly_cut_off
    print("limits for anomaly: ", lower_limit, upper_limit)
    # Generate outliers
    for outlier in series:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies
