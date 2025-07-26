import os

def rename_files_with_sequential_numbers():
    """
    Prompts the user for a directory path and renames all files in it
    to a sequential format: data_image_1.ext, data_image_2.ext, etc.
    """
    # Prompt the user to enter the directory path
    directory_path = input("Please enter the full path to the folder containing the files: ")

    try:
        # Check if the provided path is a valid directory
        if not os.path.isdir(directory_path):
            print(f"Error: The path '{directory_path}' is not a valid directory.")
            return

        # Get a list of all files in the specified directory
        filenames = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        if not filenames:
            print("No files found in the specified directory.")
            return

        print(f"Found {len(filenames)} files to rename.")

        # Loop through each file and rename it
        for i, filename in enumerate(sorted(filenames)):
            # Separate the file name and its extension
            name, extension = os.path.splitext(filename)

            # Create the new file name with a sequential number
            new_filename = f"data_image_{i + 1}{extension}"

            # Get the full path for the old and new filenames
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)

            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: '{filename}' to '{new_filename}'")

        print("\nFile renaming complete.")

    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to Use This Script ---

# 1.  Run the script from your terminal.
# 2.  When prompted, paste or type the full path to the folder and press Enter.
#
#     - On Windows, it might look like: C:\Users\YourUsername\Desktop\MyPictures
#     - On macOS or Linux, it might look like: /Users/YourUsername/Desktop/MyPictures

if __name__ == "__main__":
    rename_files_with_sequential_numbers()