project(testexecutable
            VERSION ${PROJECT_VERSION}
            DESCRIPTION "Test of the executable template"
)

set(TEST_SRC 
    TestExample.cpp
)

AddUnitTest(${PROJECT_NAME} ${TEST_SRC})

target_link_libraries(${PROJECT_NAME}
    PUBLIC

)