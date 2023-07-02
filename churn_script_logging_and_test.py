'''
Modual for testing churn_library.py
'''
import os
import logging
import churn_library as cls
import pandas as pd

if not os.path.exists('./logs/'):
	os.makedirs('./logs/')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("""Testing import_data:
						 The file doesn't appear to have rows and columns""")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
	except FileNotFoundError as err:
		logging.error("""Testing perform_eda:
							The file wasn't found""")
		raise err

	try:
		df = perform_eda(df)

		assert os.path.exists('./images/')

		files = os.listdir("./images/")
		assert 'churn_histogram.png' in files
		assert 'customer_age_histogram.png' in files
		assert 'marital_status_bar.png' in files
		assert 'total_trans_count_density.png' in files
		assert 'correlation_heatmap.png' in files
		logging.info("Testing perform_eda: SUCCESS")

	except AssertionError as error:
		logging.error("""Testing test_eda:
						 One or more images were not""")
		raise err


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")

	except FileNotFoundError as err:
		logging.error("""Testing test_encoder_helper:
							The file wasn't found""")
		raise err

	init_rows, init_cols = df.shape

	try:
		df = encoder_helper(df, cls.CATEGORICAL_COLS)

	except KeyError as err:
		logging.error("""Testing test_encoder_helper:
							Columns not in dataframe""")
		raise err

	except AssertionError as err:
		logging.error("""Testing test_encoder_helper:
							Inputs are not correct type""")
		raise err

	try:
		encode_rows, encode_columns = df.shape

		assert encode_rows == init_rows
		assert encode_columns == init_cols + len(cls.CATEGORICAL_COLS)
		for col in cls.CATEGORICAL_COLS:
			updated_col_name = col + '_' + 'Churn'
			assert updated_col_name in df.columns
		logging.info("Testing tencoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("""Testing test_encoder_helper:
							Data Frame wrong shape or missing columns""")
		raise err


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering

	'''
	try:
		df = cls.import_data("./data/bank_data.csv")

	except FileNotFoundError as err:
		logging.error("""Testing perform_feature_engineering:
							The file wasn't found""")
		raise err

	try:
		X_train, X_test, y_train, y_test = perform_feature_engineering(df)
	except AssertionError as err:
		print("""Testing perform_feature_engineering:
                 Wrong parameter types""")
		raise err

	except KeyError as err:
		logging.error("""Testing perform_feature_engineering:
						 Expected columns not in dataframes""")
		raise err

	try:
		assert isinstance(X_train, pd.DataFrame)
		assert isinstance(X_test, pd.DataFrame)
		assert isinstance(y_train, pd.Series)
		assert isinstance(y_test, pd.Series)
		assert all([col in X_train.columns for col in cls.FEATURES_TO_KEEP])
		assert all([col in X_test.columns for col in cls.FEATURES_TO_KEEP])
		logging.info("Testing perform_feature_engineering: SUCCESS")

	except AssertionError as err:
		logging.error("""Testing perform_feature_engineering:
		 				 Outputs wrong time or missing features""")
		raise err


def test_train_models(train_models):
	'''
	test train_models
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")

	except FileNotFoundError as err:
		logging.error("""Testing train_models:
							The file wasn't found""")
		raise err

	try:
		X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
	    	df.head(100))
	except AssertionError as err:
		logging.error("""Testing test_train_models:
                		 Wrong parameter types""")
		raise err

	try:
		train_models(X_train, X_test, y_train, y_test)

		assert os.path.exists('./images/')
		assert os.path.exists('./models/')

		image_files = os.listdir("./images/")
		assert 'randomforest_report.png' in image_files
		assert 'logisticregression_report.png' in image_files
		assert 'feature_importance.png' in image_files
		assert 'model_roc.png' in image_files


		model_files = os.listdir("./models/")
		assert 'rfc_model.pkl' in model_files
		assert 'logistic_model.pkl' in model_files

		logging.info("Testing test_train_models: SUCCESS")
	except AssertionError as err:
		logging.error("""Testing test_train_models:
                		 Missing files from images or models""")
		raise err


if __name__ == "__main__":
	test_import(cls.import_data)
	test_eda(cls.perform_eda)
	test_encoder_helper(cls.encoder_helper)
	test_perform_feature_engineering(cls.perform_feature_engineering)
	test_train_models(cls.train_models)
