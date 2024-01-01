import os

from sklearn.model_selection import train_test_split
import pandas as pd


def combine_dataset(path:str) -> pd.DataFrame:

    # Combine both sources into a dataset
    # Specify the folder path
    folder_path = path

    # Get the list of files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over the files and read them into the dataframe
    files = []
    for file_name in file_list:
        files.append(os.path.join(folder_path, file_name))

    df_1 = pd.read_csv(files[0], sep="\t", header=None)
    df_2 = pd.read_csv(files[1], sep="\t", header=None)

    df = pd.concat([df_1, df_2])

    # reset the index
    df.reset_index(drop=True, inplace=True)

    # rename columns
    df.rename(columns={0: 'lemma', 1: 'inflected form', 2: 'PoS'}, inplace=True)

    # shuffle columns
    df = df.sample(frac=1, axis=0, random_state=123)

    return df

def spacify(x):
    return  '< ' +''.join([c + ' ' for c in x]) + '>'

def semicolons_for_spaces(x):
    return x.replace(";", " ")

def main() -> None:

    language = "dan"
    danish_df = combine_dataset("../data/sigmorphon2017")

    # Split strings in spaces and add <>, so adelantar turns into < a d e l a n t a r >
    # change all rows of column A by adding 2
    danish_df['lemma'] = danish_df['lemma'].apply(spacify)
    danish_df['inflected form'] = danish_df['inflected form'].apply(spacify)

    # In PoS tags, change ; for spaces
    danish_df['PoS'] = danish_df['PoS'].apply(semicolons_for_spaces)

    train_df, temp_df = train_test_split(danish_df, test_size=0.3, random_state=42)

    # Now, temp_df contains 30% of the original data
    # Next, split temp_df into dev and test sets (50% of the remaining data each)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Check the sizes of the resulting sets
    print("Train set size:", len(train_df))
    print("Dev set size:", len(dev_df))
    print("Test set size:", len(test_df))

    # Now, we need to export the data in both the input and the output format
    for split in ["train", "dev", "tst"]:

        input_file = open(f"{split}.{language}.input", "w")
        output_file = open(f"{split}.{language}.output", "w")

        dataframe = 0

        if split == "train":
            dataframe = train_df
        elif split == "dev":
            dataframe = dev_df
        else:
            dataframe = test_df

        # Export input file
        for _,line in dataframe.iterrows():
            input_file.write(f"{line['lemma']} {line['PoS']}\n")
            output_file.write(f"{line['inflected form']}\n")

        input_file.close()
        output_file.close()
            


if __name__ == "__main__":
    main()