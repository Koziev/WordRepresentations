/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/model/feed_forward_net.h"
#include "singa/model/optimizer.h"
#include "singa/model/metric.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"

#include "singa/core/initialize_static_ctors.h" // added by inkoziev





using namespace singa;


// currently supports 'cudnn' and 'singacpp'
#ifdef USE_CUDNN
const std::string engine = "cudnn";
#else
const std::string engine = "singacpp";
#endif  // USE_CUDNN



static LayerConf GenOutputDenseConf(string name, int num_output, float std, float wd)
{
	LayerConf conf;
	conf.set_name(name);
	//conf.set_type("singacpp_dense");
	conf.set_type("singacpp_dense");
	DenseConf *dense = conf.mutable_dense_conf();
	dense->set_num_output(num_output);

	ParamSpec *wspec = conf.add_param();
	wspec->set_name(name + "_weight");
	wspec->set_decay_mult(wd);
	auto wfill = wspec->mutable_filler();
	wfill->set_type("Gaussian");
	wfill->set_std(std);

	ParamSpec *bspec = conf.add_param();
	bspec->set_name(name + "_bias");
	bspec->set_lr_mult(2);
	bspec->set_decay_mult(0);

	return conf;
}


static LayerConf GenHiddenDenseConf(string name, int num_output, float std, float wd)
{
	LayerConf conf;
	conf.set_name(name);
	conf.set_type("singacpp_dense");
	DenseConf *dense = conf.mutable_dense_conf();
	dense->set_num_output(num_output);

	ParamSpec *wspec = conf.add_param();
	wspec->set_name(name + "_weight");
	wspec->set_decay_mult(wd);
	auto wfill = wspec->mutable_filler();
	wfill->set_type("Gaussian");
	wfill->set_std(std);

	ParamSpec *bspec = conf.add_param();
	bspec->set_name(name + "_bias");
	bspec->set_lr_mult(2);
	bspec->set_decay_mult(0);

	return conf;
}

static LayerConf GenReLUConf(string name)
{
	LayerConf conf;
	conf.set_name(name);
	conf.set_type(engine + "_relu");

	return conf;
}

static LayerConf GenSigmoidConf(string name)
{
	LayerConf conf;
	conf.set_name(name);
	conf.set_type(engine + "_sigmoid");

	return conf;
}



static void split(const string &s, const char* delim, vector<string> & v)
{
	v.clear();

	// to avoid modifying original string
	// first duplicate the original string and return a char pointer then free the memory
	char * dup = strdup(s.c_str());
	char * token = strtok(dup, delim);
	while (token != NULL) {
		v.push_back(string(token));
		// the call is treated as a subsequent calls to strtok:
		// the function continues from where it left in previous invocation
		token = strtok(NULL, delim);
	}

	free(dup);
	return;
}

static Tensor load_tensor_from_csv(const std::wstring & data_folder, const wchar_t * filename, singa::DataType data_type)
{
	size_t nb_row = 0;
	size_t nb_col = 0;

	vector<string> tx;
	std::string line;
	std::ifstream rdr(data_folder + filename);

	int max_rows = 10000000;

	while (std::getline(rdr, line))
	{
		nb_row++;

		if (nb_col == 0)
		{
			split(line, "\t", tx);
			nb_col = tx.size();
		}

		if (nb_row >= max_rows) break;
	}


	size_t nb_cells = nb_row*nb_col;
	int idata = 0;

	rdr = std::ifstream(data_folder + filename);
	int row_count = 0;

	if (data_type == kFloat32)
	{
		float * data = new float[nb_cells];
		while (std::getline(rdr, line))
		{
			split(line, "\t", tx);
			for (int icol = 0; icol < nb_col; ++icol)
			{
				data[idata++] = atof(tx[icol].c_str());
			}
			row_count++;
			if (row_count >= max_rows) break;
		}

		Shape shape = Shape{ nb_row, nb_col };
		Tensor t = Tensor(shape, data_type);
		t.CopyDataFromHostPtr(data, nb_cells);
		delete[] data;

		return t;
	}
	else if (data_type == kInt)
	{
		int * data = new int[nb_cells];
		while (std::getline(rdr, line))
		{
			split(line, "\t", tx);
			for (int icol = 0; icol < nb_col; ++icol)
			{
				int y = (int)atof(tx[icol].c_str());
				data[idata++] = y;
			}
			row_count++;
			if (row_count >= max_rows) break;
		}

		Shape shape = Shape{ nb_row, nb_col };
		Tensor t = Tensor(shape, data_type);
		t.CopyDataFromHostPtr(data, nb_cells);
		delete[] data;

		return t;
	}


}

static FeedForwardNet create_net(size_t input_size)
{
	FeedForwardNet net;
	Shape s{ input_size };

	net.Add(GenHiddenDenseConf("dense1", 96, 1.0, 1.0), &s);
	net.Add(GenSigmoidConf("dense1_a"));
	net.Add(GenOutputDenseConf("dense_output", 2, 1.0, 1.0));
	net.Add(GenSigmoidConf("dense2_a"));

	return net;
}



static FeedForwardNet create_net_relu(size_t input_size)
{
	FeedForwardNet net;
	Shape s{ input_size };

	net.Add(GenHiddenDenseConf("dense1", 96, 1.0, 1.0), &s);
	net.Add(GenReLUConf("relu1_a"));
	net.Add(GenHiddenDenseConf("dense2", 96, 1.0, 1.0), &s);
	net.Add(GenReLUConf("relu2_a"));
	net.Add(GenOutputDenseConf("dense_output", 2, 1.0, 1.0));
	net.Add(GenSigmoidConf("dense2_a"));

	return net;
}


static void Train(int num_epoch, const std::wstring & data_dir)
{
	Tensor train_x = load_tensor_from_csv(data_dir, L"X_train.csv", kFloat32);
	Tensor train_y = load_tensor_from_csv(data_dir, L"y_train.csv", kInt);

	Tensor val_x = load_tensor_from_csv(data_dir, L"X_val.csv", kFloat32);
	Tensor val_y = load_tensor_from_csv(data_dir, L"y_val.csv", kInt);

	size_t nsamples = train_x.shape(0);

	CHECK_EQ(train_x.shape(0), train_y.shape(0));
	CHECK_EQ(val_x.shape(0), val_y.shape(0));

	LOG(INFO) << "Training samples = " << train_y.shape(0)
		<< ", Test samples = " << val_y.shape(0);

	auto net = create_net(train_x.shape(1));
	SGD sgd;
	OptimizerConf opt_conf;
	opt_conf.set_momentum(0.9);
	auto reg = opt_conf.mutable_regularizer();
	reg->set_coefficient(0.0000);
	sgd.Setup(opt_conf);
	sgd.SetLearningRateGenerator([](int step) {
		return 0.1;
	});

	SoftmaxCrossEntropy loss;
	//MSE loss;
	Accuracy acc;
	net.Compile(true, &sgd, &loss, &acc);
#ifdef USE_CUDNN
	auto dev = std::make_shared<CudaGPU>();
	net.ToDevice(dev);
	train_x.ToDevice(dev);
	train_y.ToDevice(dev);
	test_x.ToDevice(dev);
	test_y.ToDevice(dev);
#endif  // USE_CUDNN

	size_t batch_size = 128;
	net.Train(batch_size, num_epoch, train_x, train_y, val_x, val_y);

	LOG(INFO) << "Start evaluating";

	Tensor holdout_x = load_tensor_from_csv(data_dir, L"X_holdout.csv", kFloat32);
	Tensor holdout_y = load_tensor_from_csv(data_dir, L"y_holdout.csv", kInt);

	std::pair<Tensor, Tensor> holdout_res = net.Evaluate(holdout_x, holdout_y, 256);

	float h_loss = Sum(holdout_res.first) / holdout_y.Size();
	float h_accuracy = Sum(holdout_res.second) / holdout_y.Size();

	std::cout << "Holdout loss=" << h_loss << " accuracy=" << h_accuracy << std::endl;

	return;
}



int main(int argc, char **argv)
{
	singa::InitChannel(nullptr);

	//singa::initialize_static_ctors();
	std::vector<std::string> rl = singa::GetRegisteredLayers();

	int pos = singa::ArgPos(argc, argv, "-epoch");
	int nEpoch = 1;
	if (pos != -1) nEpoch = atoi(argv[pos + 1]);
	pos = singa::ArgPos(argc, argv, "-data");

	int nb_epochs = 100;

	std::wstring data_folder(L"e:/polygon/WordRepresentations/data/");

	LOG(INFO) << "Start training";
	Train(nb_epochs, data_folder);
	LOG(INFO) << "End training";

	return 0;
}
