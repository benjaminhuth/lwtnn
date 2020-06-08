#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <string>
#include <random>
#include <chrono>

#include <lwtnn/FittableLWTNN.hh>
#include <lwtnn/parse_json.hh>

// global random stuff for convenience
std::mt19937 gen( (std::random_device())() );
std::uniform_real_distribution<double> dist(-3.0, 3.0);

autodiff::var f(Eigen::Vector2var x)
{
    using std::cos, std::sin;
    return cos(x[0] + x[1]) /** sin(x[0] - x[1]) * sin(x[0] - x[1])*/;
}

auto generate_data(std::size_t size)
{    
    std::vector<Eigen::VectorXvar> x(size);
    std::vector<Eigen::VectorXvar> y(size);
    
    for(auto i=0ul; i<size; ++i)
    {
        Eigen::Vector2var xval;
        xval[0] = dist(gen);
        xval[1] = dist(gen);
        x[i] = xval;
        
        Eigen::VectorXvar yval(1);
        yval[0] = f(xval);
        y[i] = yval;
    }
    
    return std::make_pair(x,y);
}

int main(int argc, char **argv) 
{
    std::ifstream input_file("./simple_lwtnn_network.json");
    
    if (!input_file.is_open())
        throw std::runtime_error("could not open file 'simple_lwtnn_network.json'. Run 'keras_network.py' to create it!");
    
    auto parsed = lwt::parse_json(input_file);
        
    // randomize layers
    lwt::FittableLWTNN nn(parsed.inputs, parsed.layers, parsed.outputs);
    nn.summary();
    
    // generate train_data
    auto [x_train, y_train] = generate_data(200);
    auto [x_test,  y_test]  = generate_data(50);    
    
    double learning_rate = 0.01;
    std::size_t epochs = 250;
    std::size_t batch_size = 16;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    auto history = nn.fit(x_train, y_train, x_test, y_test, learning_rate, batch_size, epochs);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "fit took " << std::chrono::duration<double>(t1-t0).count() << " seconds" << std::endl;
    
    std::vector<int> epoch_vec(history.train_losses.size());
    std::iota(epoch_vec.begin(), epoch_vec.end(),1);
    
    std::cout << "R2 train:      " << history.train_score.back() << std::endl;
    std::cout << "R2 validation: " << history.valid_score.back() << std::endl;
      
    for( std::size_t i=0; i<x_test.size(); ++i )
    {
        const auto &x = x_test[i];
        const auto &y = y_test[i];
        
        std::map< std::string, double > nn_inputs;
        
        nn_inputs["x1"] = static_cast<double>(x[0]);
        nn_inputs["x2"] = static_cast<double>(x[1]);
        
        auto nn_outputs = nn.compute(nn_inputs);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "x: { " << x[0] << ", " << x[1] << " }\t => ";
        std::cout << "pred: { " << nn_outputs["y"] << " }\t";
        std::cout << "true: { " << y[0]            << " }";
        std::cout << std::endl;
    }
}
