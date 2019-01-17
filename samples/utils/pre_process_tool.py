import os


def get_data_path_list(data_root_path, depth=1):
    data_path_list = []
    if depth == 2:
        for index, folder in enumerate(os.listdir(data_root_path)):
            for filename in os.listdir(data_root_path + "/" + folder):
                data_path_list.append(data_root_path + "/" + folder + "/" + filename)
    elif depth == 1:
        for filename in os.listdir(data_root_path):
            data_path_list.append(data_root_path + "/" + filename)
    else:
        pass
    data_path_list = sorted(data_path_list)
    return data_path_list




