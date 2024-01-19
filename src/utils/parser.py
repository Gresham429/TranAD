import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--preprocess', 
                    action='store_true', 
                    help="preprocess the data")
parser.add_argument('--train', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
args = parser.parse_args()