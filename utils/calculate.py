import os
import glob
import json
from datetime import datetime


def get_latest_folder(base_path):
    # Get a list of all folders in the base path, name format is %Y%m%d_%H%M%S
    folders = [f for f in glob.glob(os.path.join(base_path, '*')) if os.path.isdir(f)]
    # If there are no folders, return None
    if not folders:
        return None, None
    # Sort the folders by their name, which should be in the YYYYMMDD format
    folder_list = sorted(folders, key=lambda f: datetime.strptime(os.path.basename(f), "%Y%m%d_%H%M%S"))
    return folder_list[-1].split('/')[-1], folder_list


def get_clients_uploads_after(base_path, new_version):
    clients_dict = {}
    path_list = []
    dataset_length_list = []
    # Traverse the base path to find all clients
    for client_id in os.listdir(base_path):
        client_path = os.path.join(base_path, client_id)

        if os.path.isdir(client_path):
            client_uploads = []

            # Traverse the client's folder for dates
            for date_folder in os.listdir(client_path):
                date_path = os.path.join(client_path, date_folder)
                if os.path.isdir(date_path) and date_folder > new_version.split('_')[0]:
                    # Traverse the time folders within the date folder
                    for time_folder in os.listdir(date_path):
                        time_path = os.path.join(date_path, time_folder)

                        if os.path.isdir(time_path) and "{}{}".format(date_folder, time_folder) > new_version.replace('_', ''):
                            # Check if the 'train_dataset_length.json' file exists
                            json_file_path = os.path.join(time_path, 'train_dataset_length.json')

                            if os.path.exists(json_file_path):
                                with open(json_file_path, 'r') as f:
                                    train_dataset_length = json.load(f).get('train_dataset_length')

                                # Append the relevant information to the client's uploads
                                client_uploads.append((time_path, train_dataset_length))
                                path_list.append(time_path)
                                dataset_length_list.append(train_dataset_length)
            # If there are uploads after the specified date, add them to the dict
            if client_uploads:
                clients_dict[client_id.split('_')[-1]] = client_uploads

    return clients_dict, dataset_length_list, path_list


def calculate_client_scores(clients_uploads, coefficient, weight_size=2, total_rewards=100):
    client_scores = {}

    for client_id, uploads in clients_uploads.items():
        upload_times = len(uploads)
        total_train_dataset_length = sum([upload[1] for upload in uploads])

        # Calculate the score based on the formula: upload times * weight_size + total train dataset length
        score = upload_times * weight_size + total_train_dataset_length
        client_scores[client_id] = score

    # Calculate the total score
    total_score = sum(client_scores.values())

    # Calculate the percentage for each client
    client_score_percentages = {client_id: round(score / total_score, 4) * total_rewards * coefficient for client_id, score in
                                client_scores.items()}

    return client_score_percentages
