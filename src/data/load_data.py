import pandas as pd


def load_dataset(file_path):
    """
    Load Excel dataset.

    Parameters:
        file_path (str): Path to Excel file

    Returns:
        pandas.DataFrame
    """

    try:
        df = pd.read_excel(file_path)
        print("Dataset loaded successfully.")
        print("Shape:", df.shape)

        return df

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None

    except Exception as e:
        print("Error loading dataset:", e)
        return None