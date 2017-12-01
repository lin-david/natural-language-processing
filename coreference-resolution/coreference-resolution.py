# Feel free to import any tool for this exercise
import pandas as pd
from random import choice
import numpy as np
from sklearn.linear_model import LogisticRegression


#################################################
# ================ NAIVE SOLUTION ===============
#################################################


def naive_resolver(df):
    """
    :param df: The dataframe
    :return:
    """
    for ix in df[df.pos.str.contains("PRP")].pos.index:
        if pd.notnull(df["mention_ids"][ix]) and "PRP" in df["pos"][ix]:
            df.at[ix, "mention_ids"] = str(
                get_closest_prev_antecedent(df["mention_ids"], ix))

    return df


def get_closest_prev_antecedent(d, ix, threshold=100):
    """
    :param d: dataframe for the data
    :param ix: current row index
    :param threshold: how far back should we check
    :return:
    """
    distance = 1

    while distance < threshold:
        if pd.isnull(d[ix - distance]):
            distance += 1
            continue
        else:
            # Splitting the columns with multiple labels
            # with a space.
            options = str(d[ix - distance]).split(" ")

            # Randomly choose between the possible options
            ch = choice(options)
            return str(ch)


def replace_indices(indices):
    return str(indices)[1:-1].replace(", ", " ")

#################################################
# ================ YOUR SOLUTION(s) =============
#################################################

# We expect you to implement your own models in this
# space. The solution should return (or modify) a
# data frame in the same format as the inpust.
# The only changes to the original data frame should
# be in the column "mention_id" to those words with
# a POS tag 'PRP' or 'PRP$'
def ret_last_mentions(d, ix, threshold=50):
    """
    :param d: dataframe for the data
    :param ix: current row index
    :param threshold: how far back should we check
    :return:
    """
    lst = set()

    distance = 0

    while distance < threshold:
        if (ix - distance) < 0 or pd.isnull(d["mention_ids"][ix - distance]):
            distance += 1
            continue
        else:
            # Splitting the columns with multiple labels
            # with a space.
            options = str(d["mention_ids"][ix - distance]).split(" ")
            options_entity = str(d["entity_ids"][ix - distance]).split(" ")

            for i in options:
                for j in options_entity:
                    if (i == 'None'):
                        continue
                    else:
                        lst.add((i, j))

            distance += 1

    return list(lst)

def get_train_test(df):
    lst_all = []
    lst_matching = []

    for ix in df[df.pos.str.contains("PRP")].pos.index:
        if pd.notnull(df["mention_ids"][ix]) and "PRP" in df["pos"][ix]:
            mention_id_options = df["mention_ids"].iloc[ix].split(" ")
            entity_id_options = df["entity_ids"].iloc[ix].split(" ")

            for i in mention_id_options:
                for j in entity_id_options:
                    if (i == 'None'):
                        continue
                    else:
                        mention_id = int(i)
                        entity_id = j

                        for k in (ret_last_mentions(df[["mention_ids", "entity_ids"]], ix)):
                            other_mention_id = int(k[0])

                            if mention_id != other_mention_id and ([mention_id, other_mention_id] not in lst_all):
                                lst_all.append([mention_id, other_mention_id])
                            if mention_id != other_mention_id and entity_id == k[1] and ([mention_id, other_mention_id] not in lst_matching):
                                lst_matching.append([mention_id, other_mention_id])

    alls = np.array(lst_all)
    matching = np.array(lst_matching)
    labels = np.zeros((alls.shape[0]))

    for i in range(len(lst_all)):
        if lst_all[i] in lst_matching:
            labels[i] = 1

    return alls, labels

def ret_last_mentions_no_entity(d, ix, threshold=50):
    """
    :param d: dataframe for the data
    :param ix: current row index
    :param threshold: how far back should we check
    :return:
    """
    lst = set()

    distance = 0

    while distance < threshold:
        if (ix - distance) < 0 or pd.isnull(d[ix - distance]):
            distance += 1
            continue
        else:
            # Splitting the columns with multiple labels
            # with a space.
            options = str(d[ix - distance]).split(" ")

            for i in options:
                if (i == 'None'):
                    continue
                else:
                    lst.add(i)

            distance += 1

    return list(lst)

def get_train(df):
    lst_all = []
    for ix in df[df.pos.str.contains("PRP")].pos.index:
        if pd.notnull(df["mention_ids"][ix]) and "PRP" in df["pos"][ix]:
            mention_id_options = df["mention_ids"].iloc[ix].split(" ")

            for i in mention_id_options:
                if (i == 'None'):
                    continue
                else:
                    mention_id = int(i)

                    for k in (ret_last_mentions_no_entity(df["mention_ids"], ix)):
                        other_mention_id = int(k)

                        if mention_id != other_mention_id and ([mention_id, other_mention_id] not in lst_all):
                            lst_all.append([mention_id, other_mention_id])

    alls = np.array(lst_all)

    return alls

def better_resolver(train_df, test_df):
    sln = {}
    X_train, Y_train = get_train_test(train_df)
    X_test = get_train(test_df)

    df = pd.DataFrame(columns=['mention_ids'])

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict_proba(X_test)[:,1]

    for i in range(X_test.shape[0]):
        if Y_pred[i] > .50:
            val = X_test[i]
            sln[val[0]] = str(val[1])

    for ix in test_df[test_df.pos.str.contains("PRP")].pos.index:
        if pd.notnull(test_df["mention_ids"][ix]) and "PRP" in test_df["pos"][ix]:
            mention_id_options = test_df["mention_ids"].iloc[ix].split(" ")
            for j in mention_id_options:
                mention_id = int(j)
                if (mention_id in sln.keys()):
                    df.at[ix, "mention_ids"] = sln[mention_id]

    return df

def better_resolver_2(train_df, test_df):
    sln = {}
    X_train, Y_train = get_train_test(train_df)
    X_test = get_train(test_df)

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict_proba(X_test)[:,1]

    for i in range(X_test.shape[0]):
        if Y_pred[i] > .50:
            val = X_test[i]
            sln[val[0]] = str(val[1])

    for ix in test_df[test_df.pos.str.contains("PRP")].pos.index:
        if pd.notnull(test_df["mention_ids"][ix]) and "PRP" in test_df["pos"][ix]:
            mention_id_options = test_df["mention_ids"].iloc[ix].split(" ")
            for j in mention_id_options:
                mention_id = int(j)
                if (mention_id in sln.keys()):
                    test_df.at[ix, "mention_ids"] = sln[mention_id]

    return test_df

# python3 hw7.py
def main():
    # Feel free to edit, this is only illustrative
    train_df = pd.read_csv("train.coref.data.txt", sep="\t")
    dev_df = pd.read_csv("dev.coref.data.txt", sep="\t")

    # This modifies the dataframe in place. If your function
    # writes to a new file, you could read the data to pass it
    # to check_accuracy in a data frame.
    result_df = better_resolver(train_df, dev_df)["mention_ids"]
    print("Accuracy of Model: {:.4f}".format(check_accuracy(result_df, dev_df)))

# For generating prediction csv
# def main():
#     train_df = pd.read_csv("train.coref.data.txt", sep="\t")
#     dev_df = pd.read_csv("dev.coref.data.txt", sep="\t")
#     test_df = pd.read_csv("test.coref.data.txt", sep="\t")
#
#     # This modifies the dataframe in place. If your function
#     # writes to a new file, you could read the data to pass it
#     # to check_accuracy in a data frame.
#     result_df = better_resolver_2(train_df, test_df)
#     result_df.to_csv("test_predictions.txt", sep='\t')


#################################################
# ================ DO NOT MODIFY ===============
#################################################


def get_index(entity_id, ix_to_entity_id, entity_id_to_ix, sequence_flag=False):
    """
    This function generates the mapping between entity and test ids
    This function was used to produce the test ids, and need not be
    used for this assignment

    :param entity_id: Current entity id
    :param ix_to_entity_id: Current mapping from test to entity
    :param entity_id_to_ix: Current mappings from entity to test ids
    :param sequence_flag: Whether the previous word was the same entity
    :return: The test_id for the word
    """
    # If sequence, don't generate new id
    if sequence_flag:
        cur_ix = entity_id_to_ix[entity_id]
        ix = max(cur_ix)
        return ix, ix_to_entity_id, entity_id_to_ix
    # If existing entity, add to test id list for entity
    if entity_id in entity_id_to_ix.keys():
        cur_ix = entity_id_to_ix[entity_id]
        ix = max(list(ix_to_entity_id.keys())) + 1
        cur_ix.append(ix)
        entity_id_to_ix[entity_id] = cur_ix
        ix_to_entity_id[ix] = entity_id
        return ix, ix_to_entity_id, entity_id_to_ix
    # Else, create new entry for entity
    else:
        # If no entry has been created
        if len(ix_to_entity_id.keys()) == 0:
            ix = -1
        else:
            ix = max(list(ix_to_entity_id.keys()))
        ix_to_entity_id[ix + 1] = entity_id
        entity_id_to_ix[entity_id] = [ix + 1]
        return ix + 1, ix_to_entity_id, entity_id_to_ix


def get_mention_ids(df, return_ds=False):
    df.reset_index(inplace=True, drop="Index")

    ix_to_entity_id = {}
    entity_id_to_ix = {}

    prev_entities = []
    mention_ids = []

    for row_ix in df.index:
        label = df["entity_ids"][row_ix]
        if pd.isnull(label):
            prev_entities = []
            mention_ids.append("")
            continue
        else:
            entities = label.split(" ")
            indices = []
            for entity in entities:
                if entity in prev_entities:
                    ix, ix_to_entity_id, entity_id_to_ix = get_index(entity,
                                                             ix_to_entity_id,
                                                             entity_id_to_ix,
                                                             True)
                    indices.append(ix)
                else:
                    ix, ix_to_entity_id, entity_id_to_ix = get_index(entity,
                                                             ix_to_entity_id,
                                                             entity_id_to_ix)
                    indices.append(ix)
            mention_ids.append(replace_indices(indices))
            prev_entities = entities

    if return_ds:
        return ix_to_entity_id, entity_id_to_ix
    else:
        df["mention_ids"] = pd.Series(mention_ids)
        return df

#################################################
# ================ ACCURACY METRIC ==============
#################################################


def check_common_preds(y_pred, y, ix_to_label):
    y_pred = set([ix_to_label[int(pred)] for pred in y_pred.split(" ")])
    y = set(y.split(" "))

    return len(y.intersection(y_pred)) > 0


def check_valid_record(df, y_pred, ix):
    r1 = df["entity_ids"][ix]
    r2 = y_pred[ix]
    pos = df["pos"][ix]

    prediction_limit = 3

    # Check conditions for valid prediction record
    if pd.isnull(r1) or pd.isnull(r2):
        return False
    if r1 == "" or r2 == "":
        return False
    if r1 is None or r2 is None:
        return False
    if r1 == "None" or r2 == "None":
        return False
    if "PRP" not in pos:
        return False
    if len(r2.split(" ")) > prediction_limit:
        return False
    else:
        return True


def check_accuracy(y_pred, df):
    """
    :param y_pred: pandas series for predicted mention_ids
    :param df: The data frame with original entity and mention ids
    :return: float: mean accuracy of the predictions
    """
    y = df["entity_ids"].copy()

    # build test_id to entity dictionaries and vice versa
    ix_to_label, label_to_ix = get_mention_ids(df, True)
    scores = []

    for ix in y_pred.index:
        if check_valid_record(df, y_pred, ix):
            scores.append(1.0 if check_common_preds(y_pred[ix], y[ix], ix_to_label)
                          else 0.0)
    return np.mean(scores)

if __name__ == "__main__":
    main()
