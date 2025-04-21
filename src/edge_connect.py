import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
import mlflow
import zipfile
from collections import defaultdict


class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')
        self.starting_epoch = 1
        self.starting_iteration = 0

        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    last_line = lines[-1]
                    try:
                        parts = last_line.split()
                        last_epoch = int(parts[0])
                        last_iteration = int(parts[1])
                        self.starting_epoch = last_epoch
                        self.starting_iteration = last_iteration
                    except (IndexError, ValueError):
                        print("⚠️ Failed to parse epoch from log file. Starting from epoch 1.")


    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self,epoch):
        if self.config.MODEL == 1:
            return self.edge_model.save(epoch=epoch)

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            return self.inpaint_model.save(epoch=epoch)

        else:
            
            edge_result = self.edge_model.save(epoch=epoch)[:-1]
            inpaint_result = self.inpaint_model.save(epoch=epoch)[:-1]

            return edge_result + inpaint_result +(f"JointModel_MODE_{self.config.MODEL}",)    


    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True,
            pin_memory=True
        )

        model = self.config.MODEL
        max_epoch = int(float((self.config.MAX_EPOCHS)))
        total_samples = len(self.train_dataset)

        if total_samples == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        num_batches = len(train_loader)
        iteration = self.starting_iteration  # Initialize iteration counter
        mlflow.set_tracking_uri(uri=self.config.MLFLOW_TRACKING_URI)

        # Create a new MLflow Experiment
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)

        #mlflow run_id
        run_id = self.config.MLFLOW_RUN_ID or None

        while (mlflow.start_run(run_id=run_id)):
            mlflow.set_tag("MODEL", self.config.MODEL)
            mlflow.log_param("batch_size", self.config.BATCH_SIZE)
            mlflow.log_param("learning_rate", self.config.LR)
            mlflow.log_param("model", self.config.MODEL)
            mlflow.log_param("max_epochs", self.config.MAX_EPOCHS)
            mlflow.log_param("input_size", self.config.INPUT_SIZE)
            mlflow.log_param("training_metrics_logging_interval", self.config.LOG_INTERVAL)
            mlflow.log_param("validation_metrics_logging_interval", "Eevery epoc")
            mlflow.log_param("running_validation_tests", "Eevery epoch")


            for epoch in range(self.starting_epoch, max_epoch + 1):
                print(f'\n\nTraining epoch: {epoch}')

                progbar = Progbar(total_samples, width=20, stateful_metrics=['epoch', 'iter'])

                for i, items in enumerate(train_loader):
                    iteration += 1
                    self.edge_model.train()
                    self.inpaint_model.train()

                    images, images_gray, edges, masks = self.cuda(*items)

                    logs = []  # Initialize logs for this iteration

                    # edge model
                    if model == 1:
                        outputs, gen_loss, dis_loss, model_logs = self.edge_model.process(images_gray, edges, masks)
                        precision, recall = self.edgeacc(edges * masks, outputs * masks)
                        logs.extend(model_logs)
                        logs.extend([('precision', precision.item()), ('recall', recall.item())])
                        self.edge_model.backward(gen_loss, dis_loss)

                        running_logs = defaultdict(list)
                        for log_name, log_value in model_logs:
                            running_logs[log_name].append(log_value)
                        running_logs["precision"].append(precision.item())
                        running_logs["recall"].append(recall.item())

                        if iteration % self.config.LOG_INTERVAL == 0:
                            for key, values in running_logs.items():
                                smoothed = np.mean(values)
                                mlflow.log_metric(key, smoothed, step=iteration)
                            running_logs.clear()
                    
                    # inpaint model
                    elif model == 2:
                        outputs, gen_loss, dis_loss, model_logs = self.inpaint_model.process(images, edges, masks)
                        outputs_merged = (outputs * masks) + (images * (1 - masks))
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                        mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                        logs.extend(model_logs)
                        logs.extend([('psnr', psnr.item()), ('mae', mae.item())])
                        self.inpaint_model.backward(gen_loss, dis_loss)

                        running_logs = defaultdict(list)
                        for log_name, log_value in model_logs:
                            running_logs[log_name].append(log_value)
                        running_logs["psnr"].append(psnr.item())
                        running_logs["mae"].append(mae.item())

                        if iteration % self.config.LOG_INTERVAL == 0:
                            for key, values in running_logs.items():
                                smoothed = np.mean(values)
                                mlflow.log_metric(key, smoothed, step=iteration)
                            running_logs.clear()

                    # inpaint with edge model
                    elif model == 3:
                        if True or np.random.binomial(1, 0.5) > 0:
                            outputs = self.edge_model(images_gray, edges, masks)
                            outputs = outputs * masks + edges * (1 - masks)
                        else:
                            outputs = edges

                        outputs, gen_loss, dis_loss, model_logs = self.inpaint_model.process(images, outputs.detach(), masks)
                        outputs_merged = (outputs * masks) + (images * (1 - masks))
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                        mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                        logs.extend(model_logs)
                        logs.extend([('psnr', psnr.item()), ('mae', mae.item())])
                        self.inpaint_model.backward(gen_loss, dis_loss)

                        running_logs = defaultdict(list)
                        for log_name, log_value in model_logs:
                            running_logs[log_name].append(log_value)
                        running_logs["psnr"].append(psnr.item())
                        running_logs["mae"].append(mae.item())

                        if iteration % self.config.LOG_INTERVAL == 0:
                            for key, values in running_logs.items():
                                smoothed = np.mean(values)
                                mlflow.log_metric(key, smoothed, step=iteration)
                            running_logs.clear()

                    # joint model
                    else:
                        e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                        e_outputs = e_outputs * masks + edges * (1 - masks)
                        i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                        outputs_merged = (i_outputs * masks) + (images * (1 - masks))
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                        mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                        precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                        logs.extend(e_logs)
                        logs.extend([('pre', precision.item()), ('rec', recall.item())])
                        logs.extend(i_logs)
                        logs.extend([('psnr', psnr.item()), ('mae', mae.item())])
                        self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                        self.edge_model.backward(e_gen_loss, e_dis_loss)

                        running_logs = defaultdict(list)
                        for log_name, log_value in e_logs:
                            running_logs[log_name].append(log_value)
                        for log_name, log_value in i_logs:
                            running_logs[log_name].append(log_value)
                        running_logs["psnr"].append(psnr.item())
                        running_logs["mae"].append(mae.item())
                        running_logs["precision"].append(precision.item())
                        running_logs["recall"].append(recall.item())

                        if iteration % self.config.LOG_INTERVAL == 0:
                            for key, values in running_logs.items():
                                smoothed = np.mean(values)
                                mlflow.log_metric(key, smoothed, step=iteration)
                            running_logs.clear()

                    current_batch_size = len(images)
                    batch_logs = [
                        ("epoch", epoch),
                        ("iter", iteration),
                    ] + logs

                    progbar.add(current_batch_size, values=batch_logs if self.config.VERBOSE else [x for x in batch_logs if not x[0].startswith('l_')])

                    # log model at checkpoints (still based on interval if you want)
                    if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                        self.log(batch_logs)
                    # sample model at checkpoints (still based on interval if you want)
                    if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                        self.sample()

                # Evaluate model at the end of each epoch
                print('\nstart eval...\n')
                epcoh_eval_metrics = self.eval()
                for name, val in epcoh_eval_metrics.items(): 
                    if val is not None:
                        mlflow.log_metric(name, val, step=epoch)
                
                result= self.save(epoch = epoch)
                result = list(result)

                files_to_zip = result[:-1]
                model_name = result[-1]

                epoch_number = epoch 
                zip_filename = f"{model_name}_epoch_{epoch_number}.zip"
                zip_path = os.path.join(os.path.dirname(files_to_zip[0]), zip_filename) 
                create_zip_lambda = lambda files, output: [zipfile.ZipFile(output, 'a').write(f, os.path.basename(f)) for f in files]

                create_zip_lambda(files_to_zip, zip_path)
                mlflow.log_artifact(zip_path)

                for file in files_to_zip:
                    os.remove(file)
 

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        total_precision = 0.0
        total_recall = 0.0
        total_psnr = 0.0
        total_mae = 0.0
        total_gen_loss = 0.0
        total_dis_loss = 0.0
        num_batches = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)
            batch_size = images.size(0)
            num_batches += 1

            logs = []  # Initialize logs for this batch

            # edge model
            if model == 1:
                outputs, gen_loss, dis_loss, model_logs = self.edge_model.process(images_gray, edges, masks)
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                total_precision += precision.item() * batch_size
                total_recall += recall.item() * batch_size
                total_gen_loss += gen_loss.item() * batch_size
                total_dis_loss += dis_loss.item() * batch_size

                # Append logs
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))

            # inpaint model
            elif model == 2:
                outputs, gen_loss, dis_loss, model_logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                total_psnr += psnr.item() * batch_size
                total_mae += mae.item() * batch_size
                total_gen_loss += gen_loss.item() * batch_size
                total_dis_loss += dis_loss.item() * batch_size

                # Append logs
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            # inpaint with edge model
            elif model == 3:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)
                outputs, gen_loss, dis_loss, model_logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                total_psnr += psnr.item() * batch_size
                total_mae += mae.item() * batch_size
                total_gen_loss += gen_loss.item() * batch_size
                total_dis_loss += dis_loss.item() * batch_size

                # Append logs
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            # joint model
            else:
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                total_precision += precision.item() * batch_size
                total_recall += recall.item() * batch_size
                total_psnr += psnr.item() * batch_size
                total_mae += mae.item() * batch_size
                total_gen_loss += (e_gen_loss.item() + i_gen_loss.item()) * batch_size
                total_dis_loss += (e_dis_loss.item() + i_dis_loss.item()) * batch_size

                # Append logs
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            # Update progress bar
            logs.insert(0, ("it", iteration))
            progbar.add(batch_size, values=logs)

        # Compute averages
        avg_precision = total_precision / total if total > 0 and model in [1, 4] else None
        avg_recall = total_recall / total if total > 0 and model in [1, 4] else None
        avg_psnr = total_psnr / total if total > 0 and model in [2, 3, 4] else None
        avg_mae = total_mae / total if total > 0 and model in [2, 3, 4] else None
        avg_gen_loss = total_gen_loss / total if total > 0 else None
        avg_dis_loss = total_dis_loss / total if total > 0 else None

        # Return all metrics as a dictionary
        return {
            'precision_val': avg_precision,
            'recall_val': avg_recall,
            'psnr_val': avg_psnr,
            'mae_val': avg_mae,
            'gen_loss_val': avg_gen_loss,
            'dis_loss_val': avg_dis_loss,
        }

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
