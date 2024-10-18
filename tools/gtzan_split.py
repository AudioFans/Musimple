import os
import re
import argparse
from pydub import AudioSegment

class GTZAN:
    def __init__(self, root_dir, output_dir, labels):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            output_dir (str): Output directory to save converted MP3 files.
            labels (list): List of genres in the dataset.
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.labels = labels

        # Create output directory structure for MP3 files
        self.create_output_dirs()

    def create_output_dirs(self):
        """Create directories to store train and test audio files"""
        for split in ['train', 'test']:
            for genre in self.labels:
                genre_dir = os.path.join(self.output_dir, split, genre)
                os.makedirs(genre_dir, exist_ok=True)

    def split_train_test(self, audio_names, test_fold):
        """
        Split the dataset into train and test sets based on test_fold.
        E.g., test_ids = [30, 31, 32, ..., 39].
        """
        test_audio_names = []
        train_audio_names = []

        test_ids = range(test_fold * 10, (test_fold + 1) * 10)

        for audio_name in audio_names:
            # Extract the numeric ID from the audio file name
            audio_id = int(re.search(r'\d+', audio_name).group())

            if audio_id in test_ids:
                test_audio_names.append(audio_name)
            else:
                train_audio_names.append(audio_name)

        return train_audio_names, test_audio_names

    def convert_and_save(self, file_path, target_path):
        """Convert AU format to MP3 and save to target path"""
        audio = AudioSegment.from_file(file_path, format="au")
        audio.export(target_path, format="mp3")
        print(f"Converted and saved {target_path}")

    def process_genre(self, genre, test_fold):
        """Process a single genre, split the dataset, and convert formats"""
        genre_path = os.path.join(self.root_dir, genre)
        audio_files = os.listdir(genre_path)

        # Split the dataset
        train_files, test_files = self.split_train_test(audio_files, test_fold)

        # Process training set
        for audio_name in train_files:
            file_path = os.path.join(genre_path, audio_name)
            target_path = os.path.join(self.output_dir, 'train', genre, audio_name.replace('.au', '.mp3'))
            self.convert_and_save(file_path, target_path)

        # Process test set
        for audio_name in test_files:
            file_path = os.path.join(genre_path, audio_name)
            target_path = os.path.join(self.output_dir, 'test', genre, audio_name.replace('.au', '.mp3'))
            self.convert_and_save(file_path, target_path)

    def process_dataset(self):
        """Process the entire GTZAN dataset and split it into train and test sets"""
        for idx, genre in enumerate(self.labels):
            print(f"Processing genre: {genre}...")
            test_fold = idx % 10  # Each genre has a different test_fold
            self.process_genre(genre, test_fold)


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="GTZAN Dataset Converter")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the GTZAN dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the converted MP3 files')
    args = parser.parse_args()

    # Example genre labels in the GTZAN dataset
    labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    # Initialize the GTZAN processor
    gtzan = GTZAN(args.root_dir, args.output_dir, labels)
    gtzan.process_dataset()

### how to use
# python gtzan_converter.py --root_dir /path/to/gtzan/genres --output_dir /path/to/output/directory
