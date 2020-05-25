#ifndef LWTNN_FIT_HH
#define LWTNN_FIT_HH

#include <functional>
#include <list>

#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>

#include <lwtnn/LightweightNeuralNetwork.hh>

namespace lwt 
{    
    class FittableLWTNN : public LightweightNeuralNetworkT<autodiff::var>
    {   
        std::vector<std::pair<autodiff::var *, std::size_t>> m_variables;
        
        void update_weights(const autodiff::var &loss, double learning_rate);
        
        std::string m_summary_string;
        
    public:
        struct fit_history_t
        {
            std::vector<double> train_losses;
            std::vector<double> train_score;
            std::vector<double> valid_score;
        };
        
        using vector_t = Eigen::VectorXvar;
        
        double evaluate(const std::vector<ValueMap> &test_x,
                        const std::vector<ValueMap> &test_y) const;
                        
        double evaluate(const std::vector<vector_t> &test_x,
                        const std::vector<vector_t> &test_y) const;
        
        void summary();
        
        FittableLWTNN(const std::vector<Input>& inputs,
                      const std::vector<LayerConfig>& layers,
                      const std::vector<std::string>& outputs);
        
        fit_history_t fit(const std::vector<ValueMap> &train_x,
                    const std::vector<ValueMap> &train_y,
                    const std::vector<ValueMap> &valid_x,
                    const std::vector<ValueMap> &valid_y,
                    double learning_rate,
                    std::size_t batch_size,
                    std::size_t epochs);
        
        fit_history_t fit(const std::vector<vector_t> &train_x,
                    const std::vector<vector_t> &train_y,
                    const std::vector<vector_t> &valid_x,
                    const std::vector<vector_t> &valid_y,
                    const double learning_rate,
                    const std::size_t batch_size,
                    const std::size_t epochs);
    };
}

#endif
