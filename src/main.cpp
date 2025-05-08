#include <fstream>
#include <vector>
#include <bit>
#include <format>
#include <print>

#include "ml.hpp"

struct data {
    std::unique_ptr<double[]> ptr;
    std::size_t size;
};

data read_data(const char* filename) {
    // idx3-ubyte format
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open file: {}", filename));
    }
    std::array<char, 4> magic_number_raw = {};
    std::array<char, 4> num_images_raw = {};
    std::array<char, 4> num_rows_raw = {};
    std::array<char, 4> num_cols_raw = {};
    file.read(magic_number_raw.data(), sizeof(magic_number_raw));
    file.read(num_images_raw.data(), sizeof(num_images_raw));
    file.read(num_rows_raw.data(), sizeof(num_rows_raw));
    file.read(num_cols_raw.data(), sizeof(num_cols_raw));
    std::uint32_t magic_number = std::byteswap(std::bit_cast<std::uint32_t>(magic_number_raw));
    std::uint32_t num_images = std::byteswap(std::bit_cast<std::uint32_t>(num_images_raw));
    std::uint32_t num_rows = std::byteswap(std::bit_cast<std::uint32_t>(num_rows_raw));
    std::uint32_t num_cols = std::byteswap(std::bit_cast<std::uint32_t>(num_cols_raw));
    if (magic_number != 2051) {
        throw std::runtime_error(std::format("Invalid magic number: {}", magic_number));
    }
    if (num_images == 0 || num_rows == 0 || num_cols == 0) {
        throw std::runtime_error(std::format("Invalid image dimensions: {}x{}", num_rows, num_cols));
    }
    std::size_t num_elements = num_images * num_rows * num_cols;
    std::unique_ptr<double[]> buffer = std::make_unique<double[]>(num_elements);
    char temp = 0;
    for (std::size_t i = 0; i < num_elements; ++i) {
        file.read(&temp, 1);
        buffer[i] = std::bit_cast<std::uint8_t>(temp) / 255.0; // Normalize to [0, 1]
    }
    return { std::move(buffer), num_elements };
}

struct label {
    std::unique_ptr<std::uint8_t[]> ptr;
    std::size_t size;
};

label read_labels(const char* filename) {
    // idx1-ubyte format
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open file: {}", filename));
    }
    std::array<char, 4> magic_number_raw = {};
    std::array<char, 4> num_labels_raw = {};
    file.read(magic_number_raw.data(), sizeof(magic_number_raw));
    file.read(num_labels_raw.data(), sizeof(num_labels_raw));
    std::uint32_t magic_number = std::byteswap(std::bit_cast<std::uint32_t>(magic_number_raw));
    std::uint32_t num_labels = std::byteswap(std::bit_cast<std::uint32_t>(num_labels_raw));
    if (magic_number != 2049) {
        throw std::runtime_error(std::format("Invalid magic number: {}", magic_number));
    }
    if (num_labels == 0) {
        throw std::runtime_error(std::format("Invalid number of labels: {}", num_labels));
    }
    std::unique_ptr<std::uint8_t[]> buffer = std::make_unique<std::uint8_t[]>(num_labels);
    char temp = 0;
    for (std::size_t i = 0; i < num_labels; ++i) {
        file.read(&temp, 1);
        buffer[i] = std::bit_cast<std::uint8_t>(temp);
    }
    return { std::move(buffer), num_labels };
}

constexpr std::size_t picture_size = 28 * 28;

class data_shuffle_helper {
public:
    data_shuffle_helper() = default;
    explicit data_shuffle_helper(std::span<double, picture_size> raw) noexcept : buffer(raw) {}
    data_shuffle_helper(const data_shuffle_helper& other) noexcept {
        std::ranges::copy(other.buffer, this->buffer.data());
    }

    data_shuffle_helper& operator=(const data_shuffle_helper& other) noexcept {
        if (this->buffer.data() != other.buffer.data()) {
            std::ranges::copy(other.buffer, this->buffer.data());
        }
        return *this;
    }

    void swap(data_shuffle_helper& other) noexcept {
        if (this->buffer.data() != other.buffer.data()) {
            std::ranges::swap_ranges(this->buffer, other.buffer);
        }
    }

private:
    inline static std::array<double, picture_size> default_buffer = {};
    std::span<double, picture_size> buffer = default_buffer;
};

std::vector<data_shuffle_helper> to_helper(std::span<double> raw) {
    std::vector<data_shuffle_helper> result;
    const auto size = raw.size();
    result.reserve(size / picture_size);
    for (std::size_t i = 0; i < size; i += picture_size) {
        result.emplace_back(std::span<double, picture_size>(raw.subspan(i, picture_size)));
    }
    return result;
}

int main(int argc, char** argv) {
    std::println("{}", argv[0]);

    using namespace std::literals;
    constexpr auto trd_file = "train-images.idx3-ubyte";
    constexpr auto trl_file = "train-labels.idx1-ubyte";
    constexpr auto ted_file = "t10k-images.idx3-ubyte";
    constexpr auto tel_file = "t10k-labels.idx1-ubyte";

    auto train_data = read_data(trd_file);
    auto train_label = read_labels(trl_file);
    auto test_data = read_data(ted_file);
    auto test_label = read_labels(tel_file);

    if (train_data.size != train_label.size * picture_size) {
        std::println("Train data size does not match train label size.");
        std::println("Data size: {}, Label size: {}", train_data.size, train_label.size * picture_size);
        return 3;
    }

    if (test_data.size != test_label.size * picture_size) {
        std::println("Test data size does not match test label size.");
        std::println("Data size: {}, Label size: {}", test_data.size, test_label.size * picture_size);
        return 4;
    }

    std::span<double> trd(train_data.ptr.get(), train_data.size);
    std::span<std::uint8_t> trl(train_label.ptr.get(), train_label.size);
    std::span<double> ted(test_data.ptr.get(), test_data.size);
    std::span<std::uint8_t> tel(test_label.ptr.get(), test_label.size);

    mylib::default_network<picture_size, 10> network(mylib::uninitialize);

    // std::ifstream ifile("network.bin", std::ios::binary | std::ios::in);
    // network.load(ifile);
    // std::println("Train set:");
    // network.evaluate_batch(trd, trl);
    // std::println("Test set:");
    // network.evaluate_batch(ted, tel);
    // return 0;

    network.init();
    std::random_device rd{};
    std::mt19937 gen(rd());
    auto helper = to_helper(trd);
    auto z = std::views::zip(helper, trl);
    if (helper.size() != trl.size()) {
        return 114514;
    }
    for (auto _ : std::views::iota(0, 100)) {
        std::ranges::shuffle(z, gen);
        network.fit_batch(trd, trl);
        auto accuracy = network.evaluate_batch(ted, tel);
        if (accuracy > 95.5) break;
    }

    std::ofstream file("network.bin", std::ios::binary | std::ios::out);
    if (!file) {
        std::println("Failed to open file for writing: network.bin");
        return 5;
    }

    network.store(file);

    return 0;
}
