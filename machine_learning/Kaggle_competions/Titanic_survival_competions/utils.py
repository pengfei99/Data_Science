from sklearn.model_selection import train_test_split


def split_train_test_data(feature_data, label_data):
    train_X, test_X, train_y, test_y = train_test_split(feature_data, label_data, train_size=0.7, test_size=0.3, random_state=0)
    return train_X, test_X, train_y, test_y


def split_titanic_passanger_names(full_name):
    elements = full_name.split(' ')
    surname = elements[0][:-1]
    title = elements[1][:-1]
    return surname, title


def get_titanic_passanger_surname(full_name):
    elements = full_name.split(' ')
    surname = elements[0][:-1]
    return surname


def get_titanic_passanger_title(full_name):
    elements = full_name.split(' ')
    title = elements[1][:-1]
    return title


def capitalize_first_letter(str):
    for i, c in enumerate(str):
        if not c.isdigit():
            break
    return str[:i]+str[i:].capitalize()
