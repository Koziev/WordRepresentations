// Тестовый пример решения задачи https://github.com/Koziev/WordRepresentations средствами
// библиотеки tiny-dnn

/*
Copyright (c) 2013, Taiga Nomi and the respective contributors
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"

using namespace std;
using namespace tiny_dnn;


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


static std::vector<vec_t> load_x_from_csv(const std::wstring & data_folder, const wchar_t * filename)
{
	size_t nb_row = 0;
	size_t nb_col = 0;

	vector<string> tx;
	std::string line;
	std::wstring filepath = data_folder + filename;
	std::ifstream rdr(filepath);

	int max_rows = 100000000; // для отладки

	while (std::getline(rdr, line))
	{
		nb_row++;

		if (nb_col == 0)
		{
			split(line, "\t", tx);
			nb_col = tx.size();
		}

		if (nb_row > max_rows) break;
	}

	std::vector<vec_t> result;

	rdr = std::ifstream(data_folder + filename);
	int row_count = 0;
	while (std::getline(rdr, line))
	{
		split(line, "\t", tx);
		vec_t row;
		for (int icol = 0; icol < nb_col; ++icol)
		{
			row.push_back( atof(tx[icol].c_str()) );
		}
		result.push_back(row);
		row_count++;
		if (row_count > max_rows) break;
	}

	return result;
}

static std::vector<label_t> load_y_from_csv(const std::wstring & data_folder, const wchar_t * filename)
{
	std::vector<vec_t> x = load_x_from_csv(data_folder, filename);

	std::vector<label_t> y;
	for each (auto xi in x)
	{
		y.push_back((size_t)xi[0]);
	}

	return y;
}



int main(int argc, char** argv)
{
	try
	{
		std::wstring data_folder(L"e:/polygon/WordRepresentations/data/");

		std::vector<vec_t> X_train = load_x_from_csv(data_folder, L"X_train.csv");
		std::vector<vec_t> X_val = load_x_from_csv(data_folder, L"X_val.csv");

		std::vector<label_t> y_train = load_y_from_csv(data_folder, L"y_train.csv");
		std::vector<label_t> y_val = load_y_from_csv(data_folder, L"y_val.csv");

		const size_t input_size = X_train[0].size();
		const size_t num_hidden_units = X_train[0].size();
		const size_t output_size = 2;

		const int nb_epochs = 100;
		const size_t batch_size = 128;

		auto nn = make_mlp<sigmoid_layer>({ input_size, num_hidden_units, output_size });

		gradient_descent optimizer;

		optimizer.alpha = 0.1;

		progress_display disp(X_train.size());
		timer t;

		// create callback
		int epoch_count=0;
		auto on_enumerate_epoch = [&]() {
			epoch_count++;
			std::cout << "epoch #" << epoch_count << " elapsed_time=" << t.elapsed() << "s" << std::endl;

			tiny_dnn::result res = nn.test(X_val, y_val);

			float val_acc = res.num_success / float(res.num_total);
			std::cout << "alpha=" << optimizer.alpha << "\tval_acc=" << val_acc << std::endl;

			// TODO: надо как-то делать early stopping и model checkpoint, то есть сохранять
			// текущие лучшие варианты весов и прерывать обучение, если валидация не улучшается
			// на протяжении более чем 10-20 эпох.

			optimizer.alpha *= 0.85;  // decay learning rate
			optimizer.alpha = std::max((tiny_dnn::float_t)0.00001, optimizer.alpha);

			disp.restart(X_val.size());
			t.restart();
		};

		auto on_enumerate_data = [&]() { ++disp; };

		nn.train<mse>(optimizer, X_train, y_train, batch_size, nb_epochs, on_enumerate_data,
			on_enumerate_epoch);

		std::cout << "Finished.";
	}
	catch (const nn_error& e) {
		std::cout << e.what() << std::endl;
	}

	return 0;
}



