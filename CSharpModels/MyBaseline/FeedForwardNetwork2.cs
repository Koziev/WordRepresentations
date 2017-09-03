using System;

namespace FeedForwardNetworks
{
    // Реализация vanilla MLP.
    class FeedForwardNetwork2
    {
        private int ni, nh, no;

        public float[,] wih;
        public float[,] wih_delta;
        private float[] vh;
        private float[] h_dE_ds;

        public float[,] who;
        public float[,] who_delta;
        private float[] vo;
        private float[] o_dE_ds;
        private bool is_softmax;

        public float LearningRate { get; set; } = 0.1f;
        public int BatchSize { get; set; } = 100;

        private float Nonlin(float x)
        {
            return (float)Math.Tanh(x);
        }

        private float Nonlin_Softmax(float x)
        {
            return (float)Math.Exp(-x);
        }


        private float Deriv(float f)
        {
            // производная для tanh(x)
            return (1.0f - f * f);
        }

        public FeedForwardNetwork2(bool softmax, int input_size, int hidden_size, int output_size)
        {
            ni = input_size;
            nh = hidden_size;
            no = output_size;
            is_softmax = softmax;

            Random rng = new Random();

            #region HiddenLayer
            wih = new float[input_size + 1, hidden_size];
            wih_delta = new float[input_size + 1, hidden_size];
            vh = new float[hidden_size];
            h_dE_ds = new float[hidden_size];

            for (int h = 0; h < hidden_size; ++h)
            {
                for (int i = 0; i < input_size + 1; ++i)
                {
                    wih[i, h] = 0.01f - 0.02f * (float)rng.NextDouble();
                }
            }
            #endregion HiddenLayer

            #region OutputLayer
            who = new float[hidden_size + 1, output_size];
            who_delta = new float[hidden_size + 1, output_size];
            vo = new float[output_size];
            o_dE_ds = new float[output_size];

            for (int o = 0; o < output_size; ++o)
            {
                for (int h = 0; h < hidden_size + 1; ++h)
                {
                    who[h, o] = 0.01f - 0.02f * (float)rng.NextDouble();
                }
            }
            #endregion OutputLayer
        }

        public void BeginBatch()
        {
            Array.Clear(wih_delta, 0, wih_delta.Length);
            Array.Clear(who_delta, 0, who_delta.Length);
        }

        public void EndBatch()
        {
            for (int o = 0; o < no; ++o)
            {
                for (int h = 0; h < nh; ++h)
                {
                    who[h, o] += who_delta[h, o];
                }

                // отдельно коррекция для bias'а
                who[nh, o] += who_delta[nh, o];
            }

            for (int h = 0; h < nh; ++h)
            {
                for (int i = 0; i < ni; ++i)
                {
                    wih[i, h] += wih_delta[i, h];
                }

                // отдельно корреция для bias
                wih[ni, h] += wih_delta[ni, h];
            }
        }

        public void ForwardPropagation(float[] vi)
        {
            for (int h = 0; h < nh; ++h)
            {
                float s = wih[ni, h]; // bias
                for (int i = 0; i < ni; ++i)
                {
                    s += vi[i] * wih[i, h];
                }

                vh[h] = Nonlin(s);
            }


            if (is_softmax)
            {
                float denom_a = 0;
                for (int o = 0; o < no; ++o)
                {
                    float s = who[nh, o]; // bias
                    for (int h = 0; h < nh; ++h)
                    {
                        s += vh[h] * who[h, o];
                    }

                    vo[o] = Nonlin_Softmax(s);
                    denom_a += vo[o];
                }

                for (int o = 0; o < no; ++o)
                {
                    vo[o] /= denom_a;
                }
            }
            else
            {
                for (int o = 0; o < no; ++o)
                {
                    float s = who[nh, o]; // bias
                    for (int h = 0; h < nh; ++h)
                    {
                        s += vh[h] * who[h, o];
                    }

                    vo[o] = Nonlin(s);
                }
            }

            return;
        }

        public void ForwardFromHidden(float[] vh_input)
        {
            for (int h = 0; h < nh; ++h)
            {
                vh[h] = vh_input[h];
            }


            if (is_softmax)
            {
                float denom_a = 0;

                for (int o = 0; o < no; ++o)
                {
                    float s = who[nh, o]; // bias
                    for (int h = 0; h < nh; ++h)
                    {
                        s += vh[h] * who[h, o];
                    }

                    vo[o] = Nonlin_Softmax(s);
                    denom_a += vo[o];
                }

                for (int o = 0; o < no; ++o)
                {
                    vo[o] /= denom_a;
                }
            }
            else
            {
                for (int o = 0; o < no; ++o)
                {
                    float s = who[nh, o]; // bias
                    for (int h = 0; h < nh; ++h)
                    {
                        s += vh[h] * who[h, o];
                    }

                    vo[o] = Nonlin(s);
                }
            }

            return;
        }

        public void BackwardPropagation(float[] vi, float[] goal)
        {
            // ----------------
            // dE/ds
            // ----------------
            #region OutputLayer
            if (is_softmax)
            {
                for (int o = 0; o < no; ++o)
                {
                    o_dE_ds[o] = goal[o] - vo[o];
                }
            }
            else
            {
                for (int o = 0; o < no; ++o)
                {
                    o_dE_ds[o] = (vo[o] - goal[o]) * Deriv(vo[o]);
                }
            }
            #endregion OutputLayer

            #region Hiddenlayer
            for (int h = 0; h < nh; ++h)
            {
                float dE_dy = 0;
                for (int o = 0; o < no; ++o)
                {
                    dE_dy += who[h, o] * o_dE_ds[o];
                }

                h_dE_ds[h] = dE_dy * Deriv(vh[h]);
            }
            #endregion Hiddenlayer


            // -------------------------
            // Weight correction
            // -------------------------

            float alpha = LearningRate;

            #region OutputLayer
            for (int o = 0; o < no; ++o)
            {
                float dE_ds = o_dE_ds[o];
                for (int h = 0; h < nh; ++h)
                {
                    float dw = -alpha * dE_ds * vh[h]; // TODO: gradient clipping
                    who_delta[h, o] += dw;
                }

                // отдельно коррекция для bias'а
                who_delta[nh, o] += -alpha * dE_ds; // TODO: gradient clipping
            }
            #endregion OutputLayer

            #region HiddenLayer
            for (int h = 0; h < nh; ++h)
            {
                float dE_ds = h_dE_ds[h];
                for (int i = 0; i < ni; ++i)
                {
                    float dw = -alpha * dE_ds * vi[i];
                    wih_delta[i, h] += dw;
                }

                // отдельно корреция для bias
                wih_delta[ni, h] += -alpha * dE_ds;
            }
            #endregion HiddenLayer

            return;
        }

        public float GetOutput(int index) { return vo[index]; }
        public float GetHidden(int index) { return vh[index]; }

        public float GetInputHiddenWeight( int i, int h ) { return wih[i,h]; }

        public float Loss_SumOfSquares(float[] output)
        {
            float e = 0.0f;

            for (int i = 0; i < no; ++i)
            {
                float d = (vo[i] - output[i]);
                e += d * d;
            }

            return (float)Math.Sqrt(e);
        }

        public int GetBestOutput()
        {
            float max_a = vo[0];
            int best_index = 0;

            for (int i = 1; i < no; ++i)
            {
                if (vo[i] > max_a)
                {
                    max_a = vo[i];
                    best_index = i;
                }
            }

            return best_index;
        }
    }

}
