#include <lwtnn/LightweightGraph.hh>
#include <lwtnn/FittableLWTNN.hh>

#include <omp.h>

#include <iostream>
#include <random>

namespace
{
    template<typename T>
    using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    
    template<typename T>
    using ArrayX = Eigen::Array<T, Eigen::Dynamic, 1>;
  
    template<typename T>
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    
    template<typename T>
    auto mean_squared_error(const std::vector<VectorX<T>> &pred_vals, const std::vector<VectorX<T>> &true_vals) -> T
    {
        assert(( pred_vals.size() == true_vals.size() ));    
        
        VectorX<T> squares(pred_vals.size());
        
        for(auto i=0ul; i<pred_vals.size(); ++i)
            squares[i] = (pred_vals[i] - true_vals[i]).transpose() * (pred_vals[i] - true_vals[i]);
        
        return squares.sum() / squares.size();
    }
    
    template<typename T>
    auto r_squared(const ArrayX<T> &pred_vals, const ArrayX<T> &true_vals) -> T
    {
        assert(( pred_vals.size() == true_vals.size() ));   
        
        auto mean_vals = ArrayX<T>::Ones(true_vals.size()) * ( true_vals.sum() / true_vals.size() );
        
        T nominator   = ( ( true_vals - pred_vals ) * ( true_vals - pred_vals ) ).sum();
        T denominator = ( ( true_vals - mean_vals ) * ( true_vals - mean_vals ) ).sum();
        
        using std::max;
        return max( static_cast<T>(0.0), static_cast<T>(1.0 - nominator/denominator) );
    }
    
    template<typename T>
    auto r_squared(const std::vector<VectorX<T>> &pred_vals, const std::vector<VectorX<T>> &true_vals) -> T
    {
        ArrayX<T> pred_array(pred_vals.size()), true_array(true_vals.size());
        
        for(auto i=0ul; i<pred_vals.size(); ++i)
            pred_array[i] = pred_vals[i][0];
        
        for(auto i=0ul; i<true_vals.size(); ++i)
            true_array[i] = true_vals[i][0];
        
        return r_squared(pred_array, true_array);
    }
}


namespace lwt
{
    FittableLWTNN::FittableLWTNN(const std::vector<Input>& inputs,
                                 const std::vector<LayerConfig>& layers,
                                 const std::vector<std::string>& outputs) :
        LightweightNeuralNetworkT<autodiff::var>(inputs, layers, outputs)
    {
        m_summary_string.append("Layer structure:\n");
        
        int omp_threads = 0;
        #pragma omp parallel
        {
            omp_threads = omp_get_num_threads();
        }
        std::cout << "lwtnn working with " << omp_threads << " OpenMP threads" << std::endl;
        
        // collect pointers to all variables for later update... its a bit hacky
        for(auto layer_ptr : m_stack->m_layers)  // works because of friend :)
        {
            if( auto bias_layer = dynamic_cast<BiasLayerT<autodiff::var> *>(layer_ptr) )
            {
                m_summary_string.append("\tbias layer (" + std::to_string(bias_layer->m_bias.size()) + ")\n");
                m_variables.push_back({ bias_layer->m_bias.data(), bias_layer->m_bias.size() });
            }
            else if( auto mat_layer = dynamic_cast<MatrixLayerT<autodiff::var> *>(layer_ptr) )
            {
                m_summary_string.append(
                    "\tmatrix layer (" + 
                    std::to_string(mat_layer->m_matrix.rows()) + 
                    "x" + 
                    std::to_string(mat_layer->m_matrix.cols()) + 
                    ")\n"
                );
                m_variables.push_back({ mat_layer->m_matrix.data(), mat_layer->m_matrix.size() });
            }
            else if( auto maxout = dynamic_cast<MaxoutLayerT<autodiff::var> *>(layer_ptr) )
            {
                m_summary_string.append("\tmaxout layer\n");
                m_variables.push_back({ maxout->m_bias.data(), maxout->m_bias.size() });
                for( auto &mat : maxout->m_matrices )
                    m_variables.push_back({ mat.data(), mat.size() });
            }
            else if( auto norm_layer = dynamic_cast<NormalizationLayerT<autodiff::var> *>(layer_ptr) )
            {
                m_summary_string.append("\tnormalization layer\n");
                m_variables.push_back({ norm_layer->_b.data(), norm_layer->_b.size() });
                m_variables.push_back({ norm_layer->_W.data(), norm_layer->_W.size() });
            }
            else if( auto high_layer = dynamic_cast<HighwayLayerT<autodiff::var> *>(layer_ptr) )
            {
                m_summary_string.append("\thighway layer\n");
                m_variables.push_back({ high_layer->m_b_c.data(), high_layer->m_b_c.size() });
                m_variables.push_back({ high_layer->m_b_t.data(), high_layer->m_b_t.size() });
                
                m_variables.push_back({ high_layer->m_w_c.data(), high_layer->m_w_c.size() });
                m_variables.push_back({ high_layer->m_w_t.data(), high_layer->m_w_t.size() });
            }
            else if( auto unary_act_layer = dynamic_cast<UnaryActivationLayerT<autodiff::var> *>(layer_ptr) )
            {
                m_summary_string.append("\tactivation layer\n");
            }
            else if( auto softmax_layer = dynamic_cast<SoftmaxLayerT<autodiff::var> *>(layer_ptr) )
            {
                m_summary_string.append("\tactivation layer\n");
            }
            else
            {
                throw std::runtime_error("got unexpected layer");
            }
        }
            
        auto n_var = std::accumulate(m_variables.begin(), m_variables.end(), 0, [&](auto a, auto b){ return a + b.second; });
        m_summary_string.append("-----------------------------------------\n");
        m_summary_string.append("total variables: " + std::to_string(n_var));
    }
    
    void FittableLWTNN::summary()
    {
        std::cout << m_summary_string << std::endl;
    }
    
    void FittableLWTNN::update_weights(const autodiff::var &loss, double learning_rate)
    {
        #pragma omp parallel for
        for(auto i=0ul; i<m_variables.size(); ++i )
        {
            auto [ptr, size] = m_variables[i];
            
            for(auto j=0ul; j<size; ++j)
            {
                auto dL_dw = autodiff::derivatives(loss, autodiff::wrt(ptr[j]))[0];
                
                ptr[j] = static_cast<double>(ptr[j]) - learning_rate * dL_dw;
            }
        }
    }
    
    FittableLWTNN::fit_history_t FittableLWTNN::fit(const std::vector<ValueMap> &train_x,
                                                    const std::vector<ValueMap> &train_y,
                                                    const std::vector<ValueMap> &valid_x,
                                                    const std::vector<ValueMap> &valid_y,
                                                    double learning_rate,
                                                    std::size_t batch_size,
                                                    std::size_t epochs)
    {        
        const auto &preproc = *m_preproc;
        std::vector<vector_t> pp_train_x, pp_train_y, pp_valid_x, pp_valid_y;
        
        std::vector< std::pair<const std::vector<ValueMap> &, std::vector<vector_t> &> > table = 
        {
            { train_x, pp_train_x },
            { train_y, pp_train_y },
            { valid_x, pp_valid_x },
            { valid_y, pp_valid_y }            
        };
        
        for(auto [vm, ppv] : table)
            for(auto el : vm)
                ppv.push_back( preproc(el) );
            
        return fit(pp_train_x, pp_train_y, pp_valid_x, pp_valid_y, learning_rate, batch_size, epochs);
    }
    
    double FittableLWTNN::evaluate(const std::vector<ValueMap> &test_x,
                                   const std::vector<ValueMap> &test_y) const
    {
        const auto &preproc = *m_preproc;
        std::vector<vector_t> pp_test_x, pp_test_y;
        
        std::vector< std::pair<const std::vector<ValueMap> &, std::vector<vector_t> &> > table = 
        {
            { test_x, pp_test_x },
            { test_y, pp_test_y }           
        };
        
        for(auto [vm, ppv] : table)
            for(auto el : vm)
                ppv.push_back( preproc(el) );
            
        return evaluate(pp_test_x, pp_test_y);
    }
    
    double FittableLWTNN::evaluate(const std::vector<vector_t> &test_x,
                                   const std::vector<vector_t> &test_y) const
    {
        assert(test_x.size() == test_y.size());
        
        Eigen::ArrayXd true_vals(test_x.size()), pred_vals(test_x.size());
            
        #pragma omp parallel for
        for(auto i=0ul; i <test_x.size(); ++i)
        {
            pred_vals[i] = static_cast<double>(m_stack->compute(test_x[i])[0]);
            true_vals[i] = static_cast<double>(test_y[i][0]);
        }
        
        return r_squared(pred_vals, true_vals);
    }
    
    FittableLWTNN::fit_history_t FittableLWTNN::fit(const std::vector<vector_t> &train_x,
                                                    const std::vector<vector_t> &train_y,
                                                    const std::vector<vector_t> &valid_x,
                                                    const std::vector<vector_t> &valid_y,
                                                    const double learning_rate,
                                                    const std::size_t batch_size,
                                                    const std::size_t epochs)
    {        
        assert(train_x.size() == train_y.size());
        assert(valid_x.size() == valid_y.size());
        
        fit_history_t history;
        
        std::vector<std::size_t> indices(train_x.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        for(auto e = 0ul; e<epochs; ++e)
        {            
            std::vector<double> losses;
            std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
            
            const std::size_t num_batches = indices.size()/batch_size;
            std::size_t i=0;
            
            for(std::size_t b=0; b < num_batches; ++b)
            {                
                const auto current_batch_size = std::min(batch_size, indices.size()-i);
                
                std::vector<Eigen::VectorXvar> trues(current_batch_size), preds(current_batch_size);
                
//                 #pragma omp parallel for
                for(auto j=0ul; j<current_batch_size; ++j)
                {
                    trues[j] = train_y[ indices[i] ];
                    preds[j] = m_stack->compute( train_x[ indices[i] ] );
                    ++i;
                }
                
                auto loss = mean_squared_error(preds, trues);
                losses.push_back(static_cast<double>(loss));
                
                update_weights(loss, learning_rate);
            }
            
            history.valid_score.push_back( evaluate(valid_x, valid_y) );
            history.train_score.push_back( evaluate(train_x, train_y) );
            history.train_losses.push_back( Eigen::Map<VectorXd>(losses.data(), losses.size()).sum() / losses.size() );
            
            std::cout << "epoch #" << e << ": mean loss = " << history.train_losses.back() << std::endl;
        }
        
        return history;
    }
}
