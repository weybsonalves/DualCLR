import json
import pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from sklearn.ensemble import RandomForestClassifier
from functools import partial

import os
import warnings
import sklearn
import logging
import argparse
import numpy as np

from dataset import *
from models import *


def get_run_name(args):
	run_name = f"{args.model}"
	return run_name


def train(args):
	#### Load dataset
	print(f"\nInside training function")
	print(f"\nLoading data {args.dataset}...")
	data_module = create_data_module(args)
	
	print(f"Train/Valid/Test splits of sizes {args.train_size}, {args.valid_size}, {args.test_size}")
	print(f"Num of features: {args.num_features}")

	csv_logger = CSVLogger("logs", name=f"{args.experiment_name}")


	#### Baselines training
	if args.use_crm:
		crm = CRM(input_dim=args.num_features, hidden_dim=args.crm_hidden_dim, latent_dim=args.crm_latent_dim, proj_dim=args.crm_projector_dim)
		feature_std = torch.from_numpy(np.std(data_module.X_train, axis=0)).float()
		pretrain_dataset = PretrainDataset(
			data_module.X_train, data_module.y_train, feature_std,
			p_mask=args.aug_p_mask, sigma_noise=args.aug_sigma_noise,
			p_synthetic=args.aug_p_synthetic, beta=args.aug_beta
		)
		pretrain_dataloader = DataLoader(
			pretrain_dataset, batch_size=args.batch_size, shuffle=True,
			num_workers=args.num_workers, pin_memory=args.pin_memory
		)
		pretrain_model = PretrainModel(args, crm)
		pretrain_trainer = pl.Trainer(
			max_epochs=args.pretrain_epochs,
			accelerator="auto",
			devices="auto",
			logger=csv_logger,
			callbacks=[RichProgressBar()]
		)
		pretrain_trainer.fit(pretrain_model, pretrain_dataloader)
		
		
		if args.use_supcon:
			supcon_model = SupConFineTuner(args, crm)
			trainer = pl.Trainer(max_epochs=args.supcon_epochs, gpus=1)
			trainer.fit(supcon_model, pretrain_dataloader)
		
		# Freeze CRM
		crm.requires_grad_(False)

		# Compute latent representations
		with torch.no_grad():
			h_train, _ = crm(torch.tensor(data_module.X_train))
			h_valid, _ = crm(torch.tensor(data_module.X_valid))
			h_test, _ = crm(torch.tensor(data_module.X_test))

			h_train = h_train.cpu().numpy()
			h_valid = h_valid.cpu().numpy()
			h_test = h_test.cpu().numpy()
		data_module.set_latent_representations(h_train, h_valid, h_test)
	
	if args.model == 'rf':
		# scikit-learn expects class_weights to be a dictionary
		class_weights = {}
		for i, val in enumerate(args.class_weights):
			class_weights[i] = val

		class_weights_list = [class_weights[i] for i in range(len(class_weights))]
	
		model = RandomForestClassifier(n_estimators=args.rf_n_estimators, 
					min_samples_leaf=args.rf_min_samples_leaf, max_depth=args.rf_max_depth,
					class_weight=class_weights, max_features='sqrt',
					random_state=42, verbose=True)
		model.fit(data_module.X_train, data_module.y_train)

		#### Log metrics
		y_pred_train = model.predict(data_module.X_train)
		y_pred_valid = model.predict(data_module.X_valid)
		y_pred_test = model.predict(data_module.X_test)

		y_hat_train = model.predict_proba(data_module.X_train)
		y_hat_valid = model.predict_proba(data_module.X_valid)
		y_hat_test = model.predict_proba(data_module.X_test)

		train_metrics = compute_all_metrics(args, data_module.y_train, y_pred_train, y_hat_train)
		valid_metrics = compute_all_metrics(args, data_module.y_valid, y_pred_valid, y_hat_valid)
		test_metrics = compute_all_metrics(args, data_module.y_test, y_pred_test, y_hat_test)

		res = {}
		for metrics, dataset_name in zip(
			[train_metrics, valid_metrics, test_metrics],
			["bestmodel_train", "bestmodel_valid", "bestmodel_test"]):
			for metric_name, metric_value in metrics.items():
				csv_logger.log_metrics({f"{dataset_name}/{metric_name}": metric_value})
				res[f"{dataset_name}/{metric_name}"] = [metric_value]

		pd.DataFrame(res).to_csv(f"{csv_logger.log_dir}/metrics.csv", index=False)

	#### Pytorch lightning training
	else:

		#### Set embedding size if it wasn't provided
		if args.wpn_embedding_size==-1:
			args.wpn_embedding_size = args.train_size
		if args.sparsity_gene_embedding_size==-1:
			args.sparsity_gene_embedding_size = args.train_size
		
		if args.max_steps!=-1:
			steps_per_epoch = np.floor(args.train_size / args.batch_size)
			args.max_epochs = int(np.ceil(args.max_steps / steps_per_epoch))
			print(f"Training for max_epochs = {args.max_epochs}")


		#### Create model
		model = create_model(args, data_module)


		##### Train
		checkpoint_callback = ModelCheckpoint(		# save best model for evaluation
			monitor=f'valid/cross_entropy_loss',
			mode='min',
			save_last=True,
			verbose=True
		)

		callbacks = [checkpoint_callback, RichProgressBar()]
		if args.patience_early_stopping and args.train_on_full_data==False:
			callbacks.append(EarlyStopping(
				monitor=f'valid/cross_entropy_loss',
				mode='min',
				patience=args.patience_early_stopping,
			))
		callbacks.append(LearningRateMonitor(logging_interval='step'))

		pl.seed_everything(args.seed_training, workers=True)
		trainer = pl.Trainer(
			# Training
			max_steps=args.max_steps,
			gradient_clip_val=2.5,

			# logging
			logger=csv_logger,
			log_every_n_steps = 1,
			val_check_interval = args.val_check_interval,
			callbacks = callbacks,

			# miscellaneous
			accelerator="auto",
			devices="auto",
			detect_anomaly=True
		)
		# train
		trainer.fit(model, data_module)
		
		if args.train_on_full_data:	# if we trained on full data
			checkpoint_path = checkpoint_callback.last_model_path
		else:
			checkpoint_path = checkpoint_callback.best_model_path

			print(f"\n\nBest model saved on path {checkpoint_path}\n\n")

		#### Compute metrics for the best model
		model.log_test_key = 'bestmodel_train'
		trainer.test(model, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_path)

		model.log_test_key = 'bestmodel_valid'
		trainer.test(model, dataloaders=data_module.val_dataloader(), ckpt_path=checkpoint_path)

		model.log_test_key = 'bestmodel_test'
		trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)



	print("\nExiting from train function..")
	

def parse_arguments(args=None):
	parser = argparse.ArgumentParser()

	###############		 Dataset		###############

	parser.add_argument('--dataset', type=str, required=True,
		choices=['metabric-pam50', 'metabric-dr', 'tcga-2ysurvival', 'tcga-tumor-grade',
				 'lung', 'prostate', 'toxicity', 'cll', 'smk'])


	###############		 Model			###############

	parser.add_argument('--model', type=str, choices=['mlp', 'wpfs', 'rf', 'fsnet', 'cae', 'dietnetworks'], default='wpfs')
	parser.add_argument('--feature_extractor_dims', type=int, nargs='+', default=[100, 100, 10],
						help='layer size for the feature extractor. If using a virtual layer,\
							  the first dimension must match it.')
	parser.add_argument('--layers_for_hidden_representation', type=int, default=2, 
						help='number of layers after which to output the hidden representation used as input to the decoder \
							  (e.g., if the layers are [100, 100, 10] and layers_for_hidden_representation=2, \
							  	then the hidden representation will be the representation after the two layers [100, 100])')
	parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for the main network')


	###############		 Sparsity network and sparsity regularization		###############
	parser.add_argument('--sparsity_gene_embedding_type', type=str, default='nmf',
						choices=['feature_values', 'nmf'], help='It`s applied over data preprocessed using `embedding_preprocessing`')
	parser.add_argument('--sparsity_gene_embedding_size', type=int, default=50)
	parser.add_argument('--sparsity_regularizer', action='store_true', dest='sparsity_regularizer')
	parser.set_defaults(sparsity_regularizer=False)

	parser.add_argument('--sparsity_regularizer_hyperparam', type=float, default=0,
						help='The weight of the sparsity regularizer (used to compute total_loss)')


	###############		Weight predictor network		###############
	parser.add_argument('--wpn_embedding_type', type=str, default='nmf',
						choices=['histogram', 'feature_values', 'nmf', 'svd'],
						help='histogram = histogram x means (like FsNet)\
							  feature_values = randomly pick patients and use their gene expressions as the embedding\
							  It`s applied over data preprocessed using `embedding_preprocessing`')
	parser.add_argument('--wpn_embedding_size', type=int, default=50, help='Size of the gene embedding')

	parser.add_argument('--wpn_layers', type=int, nargs='+', default=[100, 100, 100, 100], help="The list of layer sizes for the weight predictor network.")

	###############		Concrete autoencoder parameters	###############
	parser.add_argument('--concrete_anneal_iterations', type=int, default=1000,
		help='number of iterations for annealing the Concrete radnom variables (in CAE and FsNet)')


	############### 	Scikit-learn parameters			###############
	parser.add_argument('--rf_n_estimators', type=int, default=500, help='number of trees in the random forest')
	parser.add_argument('--rf_max_depth', type=int, default=5, help='maximum depth of the tree')
	parser.add_argument('--rf_min_samples_leaf', type=int, default=2, help='minimum number of samples in a leaf')

						
	####### Training
	parser.add_argument('--use_best_hyperparams', action='store_true', dest='use_best_hyperparams',
						help="True if you don't want to use the best hyperparams for a custom dataset")
	parser.set_defaults(use_best_hyperparams=False)


	parser.add_argument('--max_steps', type=int, default=10000, help='Specify the max number of steps to train.')
	parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
	parser.add_argument('--batch_size', type=int, default=10)

	parser.add_argument('--gamma', type=float, default=0, 
						help='The factor multiplied to the reconstruction error (DietNetworks and FsNet) \
							  If >0, then create a decoder with a reconstruction loss. \
							  If ==0, then dont create a decoder.')

	parser.add_argument('--patient_preprocessing', type=str, default='z_score',
						choices=['raw', 'z_score', 'minmax'],
						help='Preprocessing applied on each COLUMN of the N x D matrix, where a row contains all gene expressions of a patient.')
	parser.add_argument('--embedding_preprocessing', type=str, default='minmax',
						choices=['raw', 'z_score', 'minmax'],
						help='Preprocessing applied on each ROW of the D x N matrix, where a row contains all patient expressions for one gene.')

	
	####### Training on the entire train + validation data
	parser.add_argument('--train_on_full_data', action='store_true', dest='train_on_full_data', \
						help='Train on the full data (train + validation), leaving only `--test_split` for testing.')

	# We observe unstable results when training on the full data (because the number of epochs differs greatly from split to split)
	# We use 10% of the training data to perform early stopping.
	parser.set_defaults(train_on_full_data=False) 	


	####### Validation
	parser.add_argument('--patience_early_stopping', type=int, default=200,
						help='Set number of checks (set by *val_check_interval*) to do early stopping.\
							 It will train for at least   args.val_check_interval * args.patience_early_stopping epochs')
	parser.add_argument('--val_check_interval', type=int, default=5, 
						help='number of steps at which to check the validation')


	####### Cross-validation
	parser.add_argument('--num_repeats', type=int, default=5, help='number of times to repeat the cross-validation; each time shuffle the data')
	parser.add_argument('--cv_folds', type=int, default=5, help="Number of CV splits")
	parser.add_argument('--repeat_id', type=int, default=0, help='each repeat_id gives a different random seed for shuffling the dataset')
	parser.add_argument('--test_split', type=int, default=0, help="Index of the test split. It should be smaller than `cv_folds`")
	parser.add_argument('--valid_percentage', type=float, default=0.1, help='Percentage of training data used for validation')
	parser.add_argument('--run_repeats_and_cv', action='store_true', dest='run_repeats_and_cv')
	parser.set_defaults(run_repeats_and_cv=False)


	####### Optimization
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--class_weight', type=str, choices=['standard', 'balanced'], default='balanced', 
						help="If `standard`, all classes use a weight of 1.\
							  If `balanced`, classes are weighted inverse proportionally to their size (see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)")

	parser.add_argument('--lr_scheduler', type=str, choices=['cosine_warm_restart', 'lambda'], default='lambda')
	
	parser.add_argument('--cosine_warm_restart_eta_min', type=float, default=1e-6)
	parser.add_argument('--cosine_warm_restart_t_0', type=int, default=35)
	parser.add_argument('--cosine_warm_restart_t_mult', type=float, default=1)

	parser.add_argument('--debugging', action='store_true', dest='debugging')
	parser.set_defaults(debugging=False)
	

	# SEEDS
	parser.add_argument('--seed_model_init', type=int, default=42, help='Seed for initializing the model (to have the same weights)')
	parser.add_argument('--seed_training', type=int, default=42, help='Seed for training (e.g., batch ordering)')

	parser.add_argument('--seed_kfold', type=int, help='Seed used for doing the kfold in train/test split')
	parser.add_argument('--seed_validation', type=int, help='Seed used for selecting the validation split.')

	# Dataset loading
	parser.add_argument('--num_workers', type=int, default=1, help="number of workers for loading dataset")
	parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='dont pin memory for data loaders')
	parser.set_defaults(pin_memory=True)


	####### Logging
	parser.add_argument('--experiment_name', type=str, default='', help='Name for the experiment')


	# CRM and pretraining arguments
	parser.add_argument('--use_crm', action='store_true', default=False, help='Use CRM for dimensionality reduction with WPFS')
	parser.add_argument('--crm_hidden_dim', type=int, default=128, help='Hidden dimension for CRM')
	parser.add_argument('--crm_latent_dim', type=int, default=64, help='Latent dimension for CRM')
	parser.add_argument('--crm_projector_dim', type=int, default=32, help='Projector dimension for CRM')
	parser.add_argument('--contrastive_temperature', type=float, default=0.1, help='Temperature for NT-Xent loss')
	parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Learning rate for pretraining CRM')
	parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of epochs for pretraining CRM')
	parser.add_argument('--aug_p_mask', type=float, default=0.05, help='Probability of masking features')
	parser.add_argument('--aug_sigma_noise', type=float, default=0.1, help='Noise scale for injection')
	parser.add_argument('--aug_p_synthetic', type=float, default=0.3, help='Probability of synthetic oversampling')
	parser.add_argument('--aug_beta', type=float, default=0.1, help='Interpolation factor for synthetic oversampling')
	
	parser.add_argument('--use_supcon', action='store_true', default=False, help='Use SupCon')
	parser.add_argument("--supcon_temperature", type=float, default=0.1)
	parser.add_argument("--supcon_lr", type=float, default=1e-3)
	parser.add_argument("--supcon_epochs", type=int, default=10)

	return parser.parse_args(args)


if __name__ == "__main__":
	warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
	warnings.filterwarnings("ignore", category=pytorch_lightning.utilities.warnings.LightningDeprecationWarning)

	print("Starting...")

	logging.basicConfig(
		filename='logs_exceptions.txt',
		filemode='a',
		format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
		datefmt='%H:%M:%S',
		level=logging.DEBUG
	)

	args = parse_arguments()
	# set experiment name if not set
	if args.experiment_name=='':
		import uuid
		args.experiment_name = str(uuid.uuid4())[:8]

	#### Assert that the dataset is supported
	SUPPORTED_DATASETS = ['metabric-pam50', 'metabric-dr',
						  'tcga-2ysurvival', 'tcga-tumor-grade',
						  'lung', 'prostate', 'toxicity', 'cll', 'smk', 'your_custom_dataset']
	if args.dataset not in SUPPORTED_DATASETS:
		raise Exception(f"Dataset {args.dataset} not supported. Supported datasets are {SUPPORTED_DATASETS}")


	# set seeds
	args.seed_kfold = args.repeat_id			# repeat_id sets the kfol random seed, which in turn created a different shuffle of the dataset
	args.seed_validation = args.test_split

	if args.dataset == 'prostate' or args.dataset == 'cll':
		# `val_check_interval`` must be less than or equal to the number of the training batches
		# 	because these two datasets are small, one cannot use batch_size=16 and val_check_interval=5
		args.val_check_interval = 4


	# BEST CONFIGS FOR EACH BASELINE AND DATASET
	if args.use_best_hyperparams:
		if args.model in ['wpfs', 'fsnet', 'dietnetworks']:
			if args.dataset=='cll':
				args.wpn_embedding_size = 70
				args.sparsity_gene_embedding_size = 70
			if args.dataset=='lung':
				args.wpn_embedding_size = 20
				args.sparsity_gene_embedding_size = 20

		elif args.model=='rf':
			params = {
				'cll': (3, 3),
				'lung': (3, 2),
				'metabric-dr': (7, 2),
				'metabric-pam50': (7, 2),
				'prostate': (5, 2),
				'smk': (5, 2),
				'tcga-2ysurvival': (3, 3),
				'tcga-tumor-grade': (3, 3),
				'toxicity': (5, 3)
			}

			args.rf_max_depth, args.rf_min_samples_leaf = params[args.dataset]
		

	if args.run_repeats_and_cv:
		# Run 5 fold cross-validation with 5 repeats

		args_new = dict(json.loads(json.dumps(vars(args))))

		for repeat_id in range(args_new['num_repeats']):
			for test_split in range(args_new['cv_folds']):
				args_new['repeat_id'] = repeat_id
				args_new['test_split'] = test_split
		
				train(argparse.Namespace(**args_new))
	else:
		train(args)