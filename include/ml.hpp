#ifndef MYLIB_MACHINE_LEARNING_H
#define MYLIB_MACHINE_LEARNING_H 1

// A FNN implementation in C++

#include <concepts>
#include <cstdint>
#include <cstddef>

#include <iostream>
#include <stdexcept>
#include <memory>
#include <tuple>
#include <array>
#include <span>
#include <mdspan>

#include <ranges>
#include <random>
#include <algorithm>
#ifndef __clang__
#include <execution>
#endif

#include <cmath>
#include <print>

namespace mylib {

    template<typename T>
    concept activation = requires (const T& t) {
        std::floating_point<typename T::data_type>;
        { t.activate(std::declval<typename T::data_type>()) } noexcept -> std::floating_point;
        { t.derivative(std::declval<typename T::data_type>()) } noexcept -> std::floating_point;
    };

    template<std::floating_point DataType = double>
    struct ReLU
    {
        using data_type = DataType;
        constexpr static data_type activate(data_type x) noexcept { return x < 0 ? 0 : x; }
        constexpr static data_type derivative(data_type x) noexcept { return x < 0 ? 0 : 1; }
    };

    static_assert(activation<ReLU<>>);

    template<std::floating_point DataType = double>
    struct leaky_ReLU
    {
        using data_type = DataType;
        constexpr static data_type activate(data_type x) noexcept { return x < 0 ? -0.01 * x : x; }
        constexpr static data_type derivative(data_type x) noexcept { return x < 0 ? -0.01 : 1; }
    };

    static_assert(activation<leaky_ReLU<>>);

    template<std::floating_point DataType = double>
    struct sigmoid
    {
        using data_type = DataType;
        constexpr static data_type activate(data_type x) noexcept { return 1 / (1 + std::exp(-x)); }
        constexpr static data_type derivative(data_type x) noexcept { return activate(x) * (1 - activate(x)); }
    };

    static_assert(activation<sigmoid<>>);

    template<std::floating_point DataType = double>
    struct tanh
    {
        using data_type = DataType;
        constexpr static data_type activate(data_type x) noexcept { return std::tanh(x); }
        constexpr static data_type derivative(data_type x) noexcept { return 1 - std::tanh(x) * std::tanh(x); }
    };

    static_assert(activation<tanh<>>);

    template<std::floating_point DataType = double>
    struct myfunc
    {
        using data_type = DataType;
        constexpr static data_type activate(data_type x) noexcept { return x <= 0 ? -0.01 * x : std::log(x); }
        constexpr static data_type derivative(data_type x) noexcept { return x <= 0 ? -0.01 : 1 / x; }
    };

    static_assert(activation<myfunc<>>);

    template<typename Config>
    concept basic_config = std::floating_point<typename Config::data_type>
        && std::convertible_to<decltype(Config::input_size), std::size_t>
        && std::convertible_to<decltype(Config::output_size), std::size_t>;

    template<typename Config>
    concept layer_config = mylib::basic_config<Config>
        && mylib::activation<typename Config::activation_function_type>
        && std::convertible_to<decltype(Config::learning_rate), typename Config::data_type>;

    template<std::size_t INPUT_SIZE, std::size_t OUTPUT_SIZE>
    struct default_layer_config
    {
        using data_type = double;
        using activation_function_type = ReLU<data_type>;
        constexpr static data_type learning_rate = 0.01;
        constexpr static std::size_t input_size = INPUT_SIZE;
        constexpr static std::size_t output_size = OUTPUT_SIZE;
    };

    template<std::size_t SIZE>
    using default_hidden_layer_config = default_layer_config<SIZE, SIZE>;

    static_assert(layer_config<default_hidden_layer_config<16>>);

    template<typename Layer>
    using requirement_t = typename Layer::requirement_type;

    template<typename Layer>
    using parameter_t = typename Layer::parameter_type;

    template<typename Layer>
    using requirement_storage_t = typename Layer::requirement_storage_type;

    template<typename Layer>
    using delegate_t = typename Layer::application_delegate;

    template<layer_config Config>
    class fixed_layer : protected Config::activation_function_type
    {
    public:
        using config_type = Config;
        using data_type = typename config_type::data_type;
        using activation_funcion_type = typename config_type::activation_function_type;
        constexpr static std::size_t input_size = config_type::input_size;
        constexpr static std::size_t output_size = config_type::output_size;
        constexpr static std::size_t weight_size = input_size * output_size;
        constexpr static std::size_t bias_size = output_size;
        constexpr static data_type learning_rate = config_type::learning_rate;

        using extents_type = std::extents<std::size_t, output_size, input_size>;
        using weights_type = std::mdspan<data_type, extents_type>;
        using const_weights_type = std::mdspan<const data_type, extents_type>;
        using biases_type = std::span<data_type, output_size>;
        using const_biases_type = std::span<const data_type, output_size>;
        using input_type = std::span<data_type, input_size>;
        using const_input_type = std::span<const data_type, input_size>;
        using output_type = std::span<data_type, output_size>;
        using const_output_type = std::span<const data_type, output_size>;

        using weights_storage_type = std::array<data_type, weight_size>;
        using biases_storage_type = std::array<data_type, bias_size>;
        using mid_result_storage_type = std::array<data_type, output_size>;
        using weights_buffer_type = std::span<data_type, weight_size>;
        using biases_buffer_type = biases_type;
        using mid_result_buffer_type = std::span<data_type, output_size>;

        using parameter_type = std::tuple<weights_buffer_type, biases_buffer_type>;
        using requirement_type = std::tuple<weights_buffer_type, biases_buffer_type, mid_result_buffer_type>;
        using requirement_storage_type = std::tuple<weights_storage_type, biases_storage_type, mid_result_storage_type>;

        fixed_layer() = default;

        template<typename Gen>
        void init(Gen& gen) noexcept {
            std::normal_distribution<data_type> distw(0, std::sqrt(2.0 / input_size));
            std::ranges::generate(this->weights_storage, [&] { return distw(gen); });
            std::uniform_real_distribution<data_type> distb(-0.01, 0.01);
            std::ranges::generate(this->biases_storage, [&] { return distb(gen); });
        }

        void load(parameter_type params) noexcept {
            auto&& [weights, biases] = params;
            std::ranges::copy(weights, this->weights_storage.data());
            std::ranges::copy(biases, this->biases_storage.data());
        }

        void store(parameter_type params) const noexcept {
            auto&& [weights, biases] = params;
            std::ranges::copy(this->weights_storage, weights.data());
            std::ranges::copy(this->biases_storage, biases.data());
        }

        weights_type weights() noexcept {
            return weights_type(this->weights_storage.data(), extents_type{});
        }

        const_weights_type weights() const noexcept {
            return const_weights_type(this->weights_storage.data(), extents_type{});
        }

        biases_type biases() noexcept {
            return biases_type(this->biases_storage.data(), output_size);
        }

        const_biases_type biases() const noexcept {
            return const_biases_type(this->biases_storage.data(), output_size);
        }

        const activation_funcion_type& activation_function() const noexcept {
            return static_cast<const activation_funcion_type&>(*this);
        }

        class application_delegate
        {
        public:
            using buffer_type = std::tuple<weights_storage_type, biases_storage_type>;

            application_delegate() = delete;
            application_delegate(const application_delegate&) = default;
            application_delegate& operator=(const application_delegate&) = default;

            template<typename Gen>
            void init(Gen& gen) noexcept { this->layer->init(gen); }

            void forward(/* out */ output_type output, const_input_type input) const noexcept {
                fixed_layer& layer = *(this->layer);
                weights_type weights = layer.weights();
                biases_type biases = layer.biases();
                // trivial implementation
                // TODO: use SIMD or other optimizations
                for (std::size_t i = 0; i < output_size; ++i) {
                    data_type& sum = this->mid_result[i];
                    sum = biases[i];
                    for (std::size_t j = 0; j < input_size; ++j) {
                        sum += weights[i, j] * input[j];
                    }
                    output[i] = layer.activation_function().activate(sum);
                }
            }

            void backward(
                /* out */ input_type prev_gradient,
                const_input_type input,
                const_output_type output,
                const_output_type next_gradient) noexcept {
                fixed_layer& layer = *(this->layer);
                weights_type weights = layer.weights();
                [[maybe_unused]] biases_type biases = layer.biases();
                // trivial implementation
                // TODO: use SIMD or other optimizations
                std::ranges::fill_n(this->weights_step.data_handle(), this->weights_step.size(), 0);
                std::ranges::fill(this->biases_step, 0);
                for (std::size_t i = 0; i < output_size; ++i) {
                    const data_type gradient = next_gradient[i] * layer.activation_function().derivative(this->mid_result[i]);
                    for (std::size_t j = 0; j < input_size; ++j) {
                        prev_gradient[j] += weights[i, j] * gradient;
                        this->weights_step[i, j] -= learning_rate * gradient * input[j];
                    }
                    this->biases_step[i] -= learning_rate * gradient;
                }
            }

            void do_update() noexcept {
                fixed_layer& layer = *(this->layer);
                auto weights = layer.weights();
                auto biases = layer.biases();
                for (std::size_t i = 0; i < output_size; ++i) {
                    for (std::size_t j = 0; j < input_size; ++j) {
                        weights[i, j] += this->weights_step[i, j];
                    }
                    biases[i] += this->biases_step[i];
                }
            }

        private:
            friend fixed_layer;
            explicit application_delegate(fixed_layer* self, requirement_type buffers) noexcept :
                weights_step(std::ranges::data(std::get<0>(buffers)), extents_type{}),
                biases_step(std::get<1>(buffers)),
                mid_result(std::get<2>(buffers)),
                layer(self)
            {}

            weights_type weights_step;
            biases_type biases_step;
            mid_result_buffer_type mid_result;
            fixed_layer* layer;
        };

        application_delegate delegate(requirement_type buffers) noexcept {
            return application_delegate(this, buffers);
        }

    private:
        weights_storage_type weights_storage = {};
        biases_storage_type biases_storage = {};
    };

    template<typename PrevLayer, typename NextLayer>
    concept connectable = (NextLayer::input_size == PrevLayer::output_size)
        && std::same_as<typename PrevLayer::data_type, typename NextLayer::data_type>
        && std::constructible_from<typename NextLayer::const_input_type, typename PrevLayer::output_type>;

    namespace details {

        template<std::size_t TAKE, typename Tuple, typename Result = std::tuple<>>
        struct take_tuple_traits;

        template<std::size_t TAKE, typename T, typename... Ts, typename... ResTs>
        struct take_tuple_traits<TAKE, std::tuple<T, Ts...>, std::tuple<ResTs...>>
        {
            using type = typename take_tuple_traits<TAKE - 1, std::tuple<Ts...>, std::tuple<ResTs..., T>>::type;
        };

        template<typename T, typename... Ts, typename... ResTs>
        struct take_tuple_traits<0, std::tuple<T, Ts...>, std::tuple<ResTs...>>
        {
            using type = std::tuple<ResTs...>;
        };

        template<typename... ResTs>
        struct take_tuple_traits<0, std::tuple<>, std::tuple<ResTs...>>
        {
            using type = std::tuple<ResTs...>;
        };

        template<std::size_t TAKE, typename Tuple>
        struct skip_tuple_traits;

        template<std::size_t TAKE, typename T, typename... Ts>
        struct skip_tuple_traits<TAKE, std::tuple<T, Ts...>>
        {
            using type = typename skip_tuple_traits<TAKE - 1, std::tuple<Ts...>>::type;
        };

        template<typename T, typename... Ts>
        struct skip_tuple_traits<0, std::tuple<T, Ts...>>
        {
            using type = std::tuple<T, Ts...>;
        };

        template<>
        struct skip_tuple_traits<0, std::tuple<>>
        {
            using type = std::tuple<>;
        };

        template<std::size_t SKIP, std::size_t TAKE, typename Tuple>
        struct sub_tuple_traits
        {
            static_assert(SKIP + TAKE <= std::tuple_size_v<Tuple>, "Tuple size out of range.");
            using type = typename take_tuple_traits<TAKE, typename skip_tuple_traits<SKIP, Tuple>::type>::type;
        };

        template<std::size_t SKIP, std::size_t TAKE>
        struct sub_tuple_fn
        {
            template<typename... Ts>
                requires (SKIP + TAKE <= sizeof...(Ts))
            constexpr auto operator()(const std::tuple<Ts...>& t)
                const noexcept -> typename sub_tuple_traits<SKIP, TAKE, std::tuple<Ts...>>::type {
                return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    return std::make_tuple(std::get<Is + SKIP>(t)...);
                }(std::make_index_sequence<TAKE>{});
            }
        };

        template<std::size_t SKIP, std::size_t TAKE>
        inline constexpr sub_tuple_fn<SKIP, TAKE> sub_tuple = {};

    } // namespace mylib::details

    template<typename PrevLayer, typename NextLayer>
        requires connectable<PrevLayer, NextLayer>
    class chained_layer
    {
    public:
        using prev_layer_type = PrevLayer;
        using next_layer_type = NextLayer;
        using data_type = typename prev_layer_type::data_type; // Also next_layer_type::data_type
        constexpr static std::size_t input_size = prev_layer_type::input_size;
        constexpr static std::size_t output_size = next_layer_type::output_size;
        constexpr static std::size_t train_size = prev_layer_type::output_size; // Also next_layer_type::input_size

        using train_type = std::span<data_type, train_size>;
        using input_type = typename prev_layer_type::input_type;
        using const_input_type = typename prev_layer_type::const_input_type;
        using output_type = typename next_layer_type::output_type;
        using const_output_type = typename next_layer_type::const_output_type;

        using train_storage_type = std::array<data_type, train_size>;
        using train_buffer_type = train_type;
        using self_requirement_type = std::tuple<train_buffer_type, train_buffer_type>; // One for activation, one for gradient
        using self_requirement_storage_type = std::tuple<train_storage_type, train_storage_type>;
        using prev_layer_requirement_type = requirement_t<prev_layer_type>;
        using next_layer_requirement_type = requirement_t<next_layer_type>;
        using prev_layer_parameter_type = parameter_t<prev_layer_type>;
        using next_layer_parameter_type = parameter_t<next_layer_type>;
        constexpr static std::size_t prev_layer_parameter_size = std::tuple_size_v<prev_layer_parameter_type>;
        constexpr static std::size_t next_layer_parameter_size = std::tuple_size_v<next_layer_parameter_type>;
        using parameter_type = decltype(std::tuple_cat(
            std::declval<prev_layer_parameter_type>(),
            std::declval<next_layer_parameter_type>()
        ));
        using requirement_type = decltype(std::tuple_cat(
            std::declval<self_requirement_type>(),
            std::declval<prev_layer_requirement_type>(),
            std::declval<next_layer_requirement_type>()
        ));
        using requirement_storage_type = decltype(std::tuple_cat(
            std::declval<self_requirement_storage_type>(),
            std::declval<requirement_storage_t<prev_layer_type>>(),
            std::declval<requirement_storage_t<next_layer_type>>()
        ));

        chained_layer() = default;

        prev_layer_type& prev_layer() noexcept { return this->prev; }
        const prev_layer_type& prev_layer() const noexcept { return this->prev; }
        next_layer_type& next_layer() noexcept { return this->next; }
        const next_layer_type& next_layer() const noexcept { return this->next; }

        void load(parameter_type params) noexcept {
            auto prev_params = details::sub_tuple<0, prev_layer_parameter_size>(params);
            auto next_params = details::sub_tuple<prev_layer_parameter_size, next_layer_parameter_size>(params);
            this->prev_layer().load(prev_params);
            this->next_layer().load(next_params);
        }

        void store(parameter_type params) const noexcept {
            auto prev_params = details::sub_tuple<0, prev_layer_parameter_size>(params);
            auto next_params = details::sub_tuple<prev_layer_parameter_size, next_layer_parameter_size>(params);
            this->prev_layer().store(prev_params);
            this->next_layer().store(next_params);
        }

        class application_delegate
        {
        public:
            using buffer_type = std::tuple<train_storage_type, train_storage_type>; // Same as requirement_type
            using prev_layer_delegate_type = delegate_t<prev_layer_type>;
            using next_layer_delegate_type = delegate_t<next_layer_type>;
            constexpr static std::size_t prev_layer_requirement_size = std::tuple_size_v<prev_layer_requirement_type>;
            constexpr static std::size_t next_layer_requirement_size = std::tuple_size_v<next_layer_requirement_type>;

            application_delegate() = delete;
            application_delegate(const application_delegate&) = default;
            application_delegate& operator=(const application_delegate&) = default;

            template<typename Gen>
            void init(Gen& gen) noexcept { this->layer->init(gen); }

            void forward(/* out */ output_type output, const_input_type input) const noexcept {
                this->prev_layer_delegate.forward(this->activation, input);
                this->next_layer_delegate.forward(output, this->activation);
            }

            void backward(
                /* out */ input_type prev_gradient,
                const_input_type input,
                const_output_type output,
                const_output_type next_gradient) noexcept {
                std::ranges::fill(this->mid_gradient, 0);
                this->next_layer_delegate.backward(this->mid_gradient, this->activation, output, next_gradient);
                this->prev_layer_delegate.backward(prev_gradient, input, this->activation, this->mid_gradient);
            }

            void do_update() noexcept {
                this->prev_layer_delegate.do_update();
                this->next_layer_delegate.do_update();
            }

        private:
            friend chained_layer;
            explicit application_delegate(chained_layer* self, requirement_type buffers) noexcept :
                layer(self), mid_gradient(std::get<0>(buffers)), activation(std::get<1>(buffers)),
                prev_layer_delegate(layer->prev_layer().delegate(details::sub_tuple<2, prev_layer_requirement_size>(buffers))),
                next_layer_delegate(layer->next_layer().delegate(details::sub_tuple<2 + prev_layer_requirement_size, next_layer_requirement_size>(buffers)))
            {}

            chained_layer* layer;
            train_type mid_gradient;
            train_type activation;
            prev_layer_delegate_type prev_layer_delegate;
            next_layer_delegate_type next_layer_delegate;
        };

        template<typename Gen>
        void init(Gen& gen) noexcept {
            this->prev_layer().init(gen);
            this->next_layer().init(gen);
        }

        application_delegate delegate(requirement_type buffers) noexcept {
            return application_delegate(this, buffers);
        }

    private:
        prev_layer_type prev;
        next_layer_type next;
    };

    template<typename... Layers>
    struct connect;

    template<typename Layer, typename... Layers>
    struct connect<Layer, Layers...>
    {
        using type = typename connect<Layer, typename connect<Layers...>::type>::type;
    };

    template<typename PrevLayer, typename NextLayer>
    struct connect<PrevLayer, NextLayer>
    {
        using type = chained_layer<PrevLayer, NextLayer>;
    };

    template<typename... Layers>
    using connect_t = typename connect<Layers...>::type;

    namespace details {

        template<typename Layer>
            requires connectable<Layer, Layer>
        struct repeat_layers_impl
        {
            using layer_type = Layer;
            template<std::size_t LAYER_COUNT>
            struct repeat
            {
                static_assert(LAYER_COUNT > 0, "Need at least one layer!");
                using type = connect_t<layer_type, typename repeat<LAYER_COUNT - 1>::type>;
            };

            template<>
            struct repeat<1>
            {
                using type = layer_type;
            };
        };

        template<typename Tuple>
        struct serialization_buffer_size_impl;

        template<typename... Spans>
        struct serialization_buffer_size_impl<std::tuple<Spans...>>
        {
            static constexpr std::size_t value = (Spans::extent + ...);
        };

        template<typename Tuple>
        struct serialization_buffer_offset_impl;

        template<typename... Spans>
        struct serialization_buffer_offset_impl<std::tuple<Spans...>>
        {
            static constexpr std::array<std::size_t, sizeof...(Spans)> value = [] {
                std::array<std::size_t, sizeof...(Spans)> offsets = {};
                std::size_t offset = 0;
                std::size_t i = 0;
                ((offsets[i++] = offset, offset += Spans::extent), ...);
                return offsets;
            }();
        };

    } // namespace mylib::details

    template<std::size_t LAYER_COUNT, typename Layer>
    struct repeat_layers
    {
        // LAYER_COUNT refers to the number of activation layers
        // The number of parameter layers is LAYER_COUNT - 1
        static_assert(LAYER_COUNT >= 2, "Need at least one parameter layer!");
        using type = typename details::repeat_layers_impl<Layer>::template repeat<LAYER_COUNT - 1>::type;
    };

    template<std::size_t LAYER_COUNT, typename Layer>
    using repeat_layers_t = typename repeat_layers<LAYER_COUNT, Layer>::type;

    inline constexpr struct uninitialize_tag_t {} uninitialize = {};

    template<std::size_t SIZE, std::floating_point DataType>
    class softmax_layer
    {
    public:
        using data_type = DataType;
        constexpr static std::size_t size = SIZE;
        constexpr static std::size_t input_size = size;
        constexpr static std::size_t output_size = size;
        using buffer_type = std::span<data_type, size>;
        using const_buffer_type = std::span<const data_type, size>;
        using input_type = buffer_type;
        using const_input_type = const_buffer_type;
        using output_type = buffer_type;
        using const_output_type = const_buffer_type;
        using storage_type = std::array<data_type, size>;
        using parameter_type = std::tuple<>;
        using requirement_type = std::tuple<buffer_type>;
        using requirement_storage_type = std::tuple<storage_type>;

        template<typename Gen>
        constexpr void init(Gen& gen) const noexcept { /* noop */ }

        void load(parameter_type params) noexcept { /* noop */ }
        void store(parameter_type params) const noexcept { /* noop */ }

        class application_delegate
        {
        public:
            application_delegate() = delete;
            application_delegate(const application_delegate&) = default;
            application_delegate& operator=(const application_delegate&) = default;
            
            template<typename Gen>
            constexpr void init(Gen& gen) const noexcept { /* noop */ }

            void forward(/* out */ output_type output, const_input_type input) const noexcept {
                std::ranges::transform(input, this->buffer.begin(), [](data_type x) static noexcept { return std::exp(x); });
                const data_type sum = std::ranges::fold_left(this->buffer, 0.0, std::plus<>{});
                std::ranges::transform(this->buffer, output.begin(), [sum](data_type x) noexcept { return x / sum; });
            }

            void backward(
                /* out */ input_type prev_gradient,
                const_input_type input,
                const_output_type output,
                const_output_type next_gradient) noexcept {
                // If i == j, then prev_gradient[i] += next_gradient[i] * output[i] * (1 - output[i])
                // if i != j, then prev_gradient[i] += next_gradient[j] * output[i] * (-output[j])
                for (std::size_t i = 0; i < size; ++i) {
                    const data_type mid = next_gradient[i] * output[i];
                    for (std::size_t j = 0; j < size; ++j) {
                        prev_gradient[j] += mid * ((i == j ? 1 : 0) - output[j]);
                    }
                }
            }

            constexpr void do_update() const noexcept { /* noop */ }

        private:
            friend softmax_layer;
            explicit application_delegate(softmax_layer* self, requirement_type buffers) noexcept
                : layer(self), buffer(std::get<0>(buffers))
            {}

            softmax_layer* layer;
            buffer_type buffer;
        };

        application_delegate delegate(requirement_type buffers) noexcept {
            return application_delegate(this, buffers);
        }
    };

    template<std::size_t INPUT_SIZE, std::size_t OUTPUT_SIZE,
        std::size_t HIDDEN_SIZE, std::size_t HIDDEN_COUNT,
        std::size_t TRAIN_BATCH,
        std::floating_point DataType = double,
        mylib::activation ActivationFunc = leaky_ReLU<DataType>,
        DataType LEARNING_RATE = 0.005
    >
        requires (INPUT_SIZE >= HIDDEN_SIZE) && (HIDDEN_SIZE >= OUTPUT_SIZE) && (HIDDEN_COUNT >= 2)
    class neural_network
    {
    public:
        using data_type = DataType;
        using activation_funcion_type = ActivationFunc;
        constexpr static std::size_t input_size = INPUT_SIZE;
        constexpr static std::size_t output_size = OUTPUT_SIZE;
        constexpr static std::size_t hidden_size = HIDDEN_SIZE;
        constexpr static std::size_t hidden_count = HIDDEN_COUNT;
        constexpr static std::size_t train_batch = TRAIN_BATCH;
        constexpr static data_type learning_rate = LEARNING_RATE;
        using input_type = std::span<data_type, input_size>;
        using const_input_type = std::span<const data_type, input_size>;
        using output_type = std::span<data_type, output_size>;
        using const_output_type = std::span<const data_type, output_size>;
        using batch_input_type = std::span<data_type, input_size * train_batch>;
        using const_batch_input_type = std::span<const data_type, input_size * train_batch>;
        using batch_output_type = std::span<data_type, output_size * train_batch>;
        using const_batch_output_type = std::span<const data_type, output_size * train_batch>;
        using post_treatment_buffer_type = std::array<data_type, output_size>;
        using label_type = std::uint8_t;
        using batch_label_type = std::span<const label_type, train_batch>;

    private:
        struct input_layer_config {
            using data_type = typename neural_network::data_type;
            using activation_function_type = typename neural_network::activation_funcion_type;
            constexpr static std::size_t input_size = neural_network::input_size;
            constexpr static std::size_t output_size = neural_network::hidden_size;
            constexpr static data_type learning_rate = neural_network::learning_rate;
        };

        struct output_layer_config {
            using data_type = typename neural_network::data_type;
            using activation_function_type = typename neural_network::activation_funcion_type;
            constexpr static std::size_t input_size = neural_network::hidden_size;
            constexpr static std::size_t output_size = neural_network::output_size;
            constexpr static data_type learning_rate = neural_network::learning_rate;
        };

        struct hidden_layer_config {
            using data_type = typename neural_network::data_type;
            using activation_function_type = typename neural_network::activation_funcion_type;
            constexpr static std::size_t input_size = neural_network::hidden_size;
            constexpr static std::size_t output_size = neural_network::hidden_size;
            constexpr static data_type learning_rate = neural_network::learning_rate;
        };

    public:
        template<std::size_t SOFTMAX_SIZE>
        using softmax_layer_type = softmax_layer<SOFTMAX_SIZE, data_type>;
        using input_layer_type = fixed_layer<input_layer_config>;
        using output_layer_type = connect_t<fixed_layer<output_layer_config>, softmax_layer_type<output_size>>;
        using hidden_layer_type = repeat_layers_t<hidden_count, fixed_layer<hidden_layer_config>>;
        using layer_type = connect_t<input_layer_type, hidden_layer_type, output_layer_type>;

    private:
        using storage_type = requirement_storage_t<layer_type>;
        using requirement_type = requirement_t<layer_type>;
        using parameter_type = parameter_t<layer_type>;
        using delegate_type = delegate_t<layer_type>;
        constexpr static std::size_t serialization_buffer_size = details::serialization_buffer_size_impl<parameter_type>::value;
        using serialization_buffer_type = std::array<data_type, serialization_buffer_size>;

        static std::unique_ptr<serialization_buffer_type> get_buffer() {
            return std::make_unique<serialization_buffer_type>();
        }

        static parameter_type buffer_to_parameter(serialization_buffer_type& buffer) noexcept {
            return [&buffer]<std::size_t... Is>(std::index_sequence<Is...>) noexcept {
                constexpr auto offset = details::serialization_buffer_offset_impl<parameter_type>::value;
                return parameter_type{
                    std::tuple_element_t<Is, parameter_type>(&buffer[offset[Is]], std::tuple_element_t<Is, parameter_type>::extent)...
                };
            }(std::make_index_sequence<std::tuple_size_v<parameter_type>>{});
        }

        static data_type loss(const_output_type predict, label_type expect) noexcept {
            // cross entropy loss
            return -std::log(predict[expect]);
        }

        static void loss_derivative(/* out */ output_type gradient,
            const_output_type predict, label_type expect) noexcept {
            // cross entropy loss derivative
            // dL/dy = -1 / y_i, where i = expect
            // dL/dy_i = 0, where i != expect
            std::ranges::fill(gradient, 0);
            gradient[expect] = -1 / predict[expect];
        }

        static void check_fit_batch(std::span<data_type> data, std::span<label_type> label) noexcept(false) {
            if (data.size() != label.size() * input_size) {
                throw std::invalid_argument("Data and label size mismatch.");
            }
            if (label.size() % train_batch != 0) {
                throw std::invalid_argument("Input size is not a multiple of train_batch.");
            }
        }

        static void check_evaluate_batch(std::span<data_type> data, std::span<label_type> label) noexcept(false) {
            if (data.size() != label.size() * input_size) {
                throw std::invalid_argument("Data and label size mismatch.");
            }
        }

    public:
        neural_network() : neural_network(mylib::uninitialize) { this->init(); }
        explicit neural_network(uninitialize_tag_t) {}

        void init() noexcept {
            std::random_device rd{};
            std::mt19937_64 gen(rd());
            this->parameter_layer->init(gen);
        }

        void predict(/* out */ output_type output, const_input_type input) const noexcept {
            this->core_delegate->forward(output, input);
        }

        inline static std::array<data_type, input_size> discard = {};

        data_type fit(const_batch_input_type data, batch_label_type label) noexcept {
            // prepare batches
            std::array<const_input_type, train_batch> data_batch =
                [&data]<std::size_t... Is>(std::index_sequence<Is...>) noexcept {
                return std::array<const_input_type, train_batch>{
                    data.template subspan<Is * input_size, input_size>()...
                };
            }(std::make_index_sequence<train_batch>{});
            std::array<data_type, train_batch> total_loss = {};
            auto batch = std::views::zip(data_batch, label, *(this->delegates), total_loss);
            // train
            auto do_train = [](auto&& zip_ref) static noexcept {
                auto&& [data, label, delegate, loss_result] = zip_ref;
                std::array<data_type, output_size> output = {};
                delegate.forward(output, data);
                loss_result = loss(output, label);
                std::array<data_type, output_size> gradient = {};
                loss_derivative(gradient, output, label);
                delegate.backward(discard, data, output, gradient);
            };
            std::for_each(
                // disable parallel for clang
                #ifndef __clang__
                std::execution::par,
                #endif
                std::ranges::begin(batch), std::ranges::end(batch), do_train);
            // do update
            // sum up all into core_layer
            auto storage_for_each = [](storage_type& core_storage, storage_type& storage, auto&& func) static noexcept {
                [&] <std::size_t... Is>(std::index_sequence<Is...>) noexcept {
                    (func(std::get<Is>(core_storage), std::get<Is>(storage)), ...);
                }(std::make_index_sequence<std::tuple_size_v<storage_type>>{});
            };
            storage_for_each(*(this->core_layer), *(this->core_layer),
                [](auto&& core, auto&& _) static noexcept {
                std::ranges::fill(core, 0);
            });
            [storage_for_each, this] <std::size_t... Is>(std::index_sequence<Is...>) noexcept {
                (storage_for_each(*(this->core_layer), std::get<Is>(*(this->application_layers)),
                    [](auto&& core, auto&& sep) static noexcept {
                    for (auto&& [c, s] : std::views::zip(core, sep)) {
                        c += s;
                    }
                }), ...);
            }(std::make_index_sequence<train_batch>{});
            this->core_delegate->do_update();
            // return average loss in this batch
            return std::ranges::fold_left(total_loss, 0.0, std::plus<>{}) / train_batch;
        }

        void fit_batch(std::span<data_type> data, std::span<label_type> label) {
            check_fit_batch(data, label);
            constexpr std::size_t data_batch_size = train_batch * input_size;
            const std::size_t total_batch = label.size() / train_batch;
            for (std::size_t i = 0; i < total_batch; ++i) {
                [[maybe_unused]] data_type loss_avg = this->fit(
                    const_batch_input_type(data.subspan(i * data_batch_size, data_batch_size)),
                    batch_label_type(label.subspan(i * train_batch, train_batch))
                );
            }
        }

        struct evaluate_result {
            bool match;
            data_type loss_value;
        };
        
        evaluate_result evaluate(const_input_type data, label_type label) const noexcept {
            std::array<data_type, output_size> output = {};
            this->core_delegate->forward(output, data);
            auto i = std::ranges::max_element(output);
            return { (i - output.begin()) == label, loss(output, label) };
        }

        data_type evaluate_batch(std::span<data_type> data, std::span<label_type> label) const {
            check_evaluate_batch(data, label);
            const std::size_t total = label.size();
            std::size_t correct = 0;
            std::size_t wrong = 0;
            data_type total_loss = 0;
            for (std::size_t i = 0; i < total; ++i) {
                auto [match, loss_value] = this->evaluate(
                    const_input_type(data.subspan(i * input_size, input_size)),
                    label[i]
                );
                total_loss += loss_value;
                ++(match ? correct : wrong);
            }
            const data_type accuracy = correct * 100.0 / total;
            std::println("accuracy: {}%, avg loss: {}", accuracy, total_loss / total);
            return accuracy;
        }

        std::ostream& store(std::ostream& os) {
            auto buffer = get_buffer();
            this->parameter_layer->store(buffer_to_parameter(*buffer));
            for (data_type p : *buffer) {
                auto temp = std::bit_cast<std::array<char, sizeof(data_type)>>(p);
                os.write(temp.data(), temp.size());
            }
            return os;
        }

        std::istream& load(std::istream& is) {
            auto buffer = get_buffer();
            for (data_type& p : *buffer) {
                std::array<char, sizeof(data_type)> temp = {};
                is.read(temp.data(), temp.size());
                p = std::bit_cast<data_type>(temp);
            }
            this->parameter_layer->load(buffer_to_parameter(*buffer));
            return is;
        }

    private:
        struct delegate_lazy_maker {
            layer_type* parameter;
            storage_type* storage;
            operator delegate_type() const noexcept { return this->parameter->delegate(*(this->storage)); }
        };

        struct batch_delegate_lazy_maker {
            neural_network* self;
            operator std::array<delegate_type, train_batch>() const noexcept {
                return [self = this->self]<std::size_t... Is>(std::index_sequence<Is...>) noexcept {
                    return std::array<delegate_type, train_batch>{
                        self->parameter_layer->delegate(std::get<Is>(*(self->application_layers)))...
                    };
                }(std::make_index_sequence<train_batch>{});
            }
        };

        std::unique_ptr<std::array<delegate_type, train_batch>> get_delegates() noexcept {
            return std::make_unique<std::array<delegate_type, train_batch>>(batch_delegate_lazy_maker(this));
        }

        std::unique_ptr<layer_type> parameter_layer = std::make_unique_for_overwrite<layer_type>();
        std::unique_ptr<std::array<storage_type, train_batch>> application_layers =
            std::make_unique_for_overwrite<std::array<storage_type, train_batch>>();
        std::unique_ptr<std::array<delegate_type, train_batch>> delegates = get_delegates();
        std::unique_ptr<storage_type> core_layer = std::make_unique_for_overwrite<storage_type>();
        std::unique_ptr<delegate_type> core_delegate =
            std::make_unique<delegate_type>(delegate_lazy_maker(this->parameter_layer.get(), this->core_layer.get()));
    };

    template<std::size_t INPUT_SIZE, std::size_t OUTPUT_SIZE>
        requires (INPUT_SIZE >= 16) && (16 >= OUTPUT_SIZE)
    using default_network = neural_network<INPUT_SIZE, OUTPUT_SIZE, 16, 2, 10>;

} // namespace mylib

#endif // !MYLIB_MACHINE_LEARNING_H
