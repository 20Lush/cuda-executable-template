
#include <gtest/gtest.h>

#include "main.hpp"

using namespace lush::EXECUTABLE;

TEST(EXECUTABLE_TEMPLATE_TEST, ExampleTest) {

	const int argc_test = 2;

	char* argv_test[argc_test + 1] = {"/my/path", "my_arg"};

	bool succes = print_args(argc_test, argv_test);

	EXPECT_EQ(succes, true);
}