/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   B L A N K   A P P L I C A T I O N                                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */  
/****************************************************************************************************************/

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>

#include <stdint.h>
#include <limits.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{

    try
    {
        std::cout << "OpenNN. Blank Application." << std::endl;
        srand((unsigned int)time(NULL));

		DataSet data_set;
		data_set.set_data_file_name("e:/polygon/WordRepresentations/data/Xy_train.csv");
		data_set.load_data();
		Variables* variables_pointer = data_set.get_variables_pointer();
		variables_pointer->set(96, 1);

		NeuralNetwork neural_network;

		LossIndex loss_index;

		TrainingStrategy training_strategy;

		const Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
		const Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

		neural_network.set(96, 96, 1);

		Inputs* inputs_pointer = neural_network.get_inputs_pointer();

		inputs_pointer->set_information(inputs_information);

		Outputs* outputs_pointer = neural_network.get_outputs_pointer();

		outputs_pointer->set_information(targets_information);

		// Loss index

		loss_index.set_data_set_pointer(&data_set);
		loss_index.set_neural_network_pointer(&neural_network);

		// Training strategy

		training_strategy.set(&loss_index);

		training_strategy.get_quasi_Newton_method_pointer()->set_maximum_iterations_number(50);

		training_strategy.perform_training();


		// Финальное тестирование готовой модели
		std::cout << "Final estimation of model..." << std::endl;
		DataSet test_set;
		test_set.set_data_file_name("e:/polygon/WordRepresentations/data/Xy_holdout.csv");
		test_set.load_data();
		Matrix<double> test_x = test_set.arrange_input_data();
		Matrix<double> test_y = test_set.arrange_target_data();
		const size_t n_test_row = test_x.get_rows_number();
		const size_t nx = test_x.get_columns_number();
		const size_t ny = test_y.get_columns_number();

		int nb_hits = 0, nb_recs = 0;
		Vector<double> inputs(nx, 0.0);
		for (int itest = 0; itest < n_test_row; ++itest)
		{
			for (size_t icol = 0; icol < nx; ++icol)
			{
				inputs[icol] = test_x(itest, icol);
			}

			Vector<double> outputs = neural_network.calculate_outputs(inputs);
			double y = outputs[0];
			if (y > 0.5)
				y = 1.0;
			else
				y = 0.0;

			double target_y = test_y(itest, 0);
			if (y == target_y)
				nb_hits++;

			nb_recs++;
		}

		double acc = nb_hits / (float)nb_recs;
		std::cout << "accuracy=" << acc << std::endl;
/*
Vector<double> inputs(2, 0.0);
Vector<double> outputs(6, 0.0);

std::cout << "X Y AND OR NAND NOR XOR XNOR" << std::endl;

inputs[0] = 1.0;
inputs[1] = 1.0;

outputs = neural_network.calculate_outputs(inputs);

std::cout << inputs.calculate_binary() << " " << outputs.calculate_binary() << std::endl;
*/


        return(0);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;

        return(1);
    }

}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
