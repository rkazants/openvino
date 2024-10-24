// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"

#include <string>

#include "gtest/gtest.h"
#include "openvino/core/variant.hpp"

using namespace ov;

class DestructorTest {
public:
    DestructorTest() {
        constructorCount++;
    }

    DestructorTest(const DestructorTest& c) {
        constructorCount++;
    }

    DestructorTest(const DestructorTest&& c) {
        constructorCount++;
    }

    ~DestructorTest() {
        destructorCount++;
    }

    static size_t destructorCount;
    static size_t constructorCount;
};
size_t DestructorTest::destructorCount = 0;
size_t DestructorTest::constructorCount = 0;

class AnyTests : public ::testing::Test {
public:
    void SetUp() override {
        DestructorTest::destructorCount = 0;
        DestructorTest::constructorCount = 0;
    }
};

TEST_F(AnyTests, parameter_std_string) {
    auto parameter = Any{"My string"};
    ASSERT_TRUE(parameter.is<std::string>());
    EXPECT_EQ(parameter.as<std::string>(), "My string");
}

TEST_F(AnyTests, parameter_int64_t) {
    auto parameter = Any{int64_t(27)};
    ASSERT_TRUE(parameter.is<int64_t>());
    EXPECT_FALSE(parameter.is<std::string>());
    EXPECT_EQ(parameter.as<int64_t>(), 27);
}

struct Ship {
    Ship(const std::string& name_, const int16_t x_, const int16_t y_) : name{name_}, x{x_}, y{y_} {}
    std::string name;
    int16_t x;
    int16_t y;
};

TEST_F(AnyTests, parameter_ship) {
    {
        auto parameter = Any{Ship{"Lollipop", 3, 4}};
        ASSERT_TRUE(parameter.is<Ship>());
        Ship& ship = parameter.as<Ship>();
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
    {
        auto parameter = Any::make<Ship>("Lollipop", int16_t(3), int16_t(4));
        ASSERT_TRUE(parameter.is<Ship>());
        Ship& ship = parameter.as<Ship>();
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
    {
        auto parameter = Any::make<Ship>("Lollipop", int16_t(3), int16_t(4));
        ASSERT_TRUE(parameter.is<Ship>());
        Ship ship = parameter;
        EXPECT_EQ(ship.name, "Lollipop");
        EXPECT_EQ(ship.x, 3);
        EXPECT_EQ(ship.y, 4);
    }
}

TEST_F(AnyTests, AnyAsInt) {
    Any p = 4;
    ASSERT_TRUE(p.is<int>());
    int test = p;
    ASSERT_EQ(4, test);
}

TEST_F(AnyTests, AnyAsUInt) {
    Any p = uint32_t(4);
    ASSERT_TRUE(p.is<uint32_t>());
    ASSERT_TRUE(p.is<uint32_t>());
    uint32_t test = p;
    ASSERT_EQ(4, test);
}

TEST_F(AnyTests, AnyAsString) {
    std::string ref = "test";
    Any p = ref;
    std::string test = p;
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_EQ(ref, test);
}

TEST_F(AnyTests, AnyAsStringInLine) {
    Any p = "test";
    std::string test = p;
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_EQ("test", test);
}

TEST_F(AnyTests, IntAnyAsString) {
    Any p = 4;
    ASSERT_TRUE(p.is<int>());
    ASSERT_FALSE(p.is<std::string>());
    ASSERT_THROW(std::string test = p, ov::Exception);
    ASSERT_THROW(std::string test = p.as<std::string>(), ov::Exception);
}

TEST_F(AnyTests, StringAnyAsInt) {
    Any p = "4";
    ASSERT_FALSE(p.is<int>());
    ASSERT_TRUE(p.is<std::string>());
    ASSERT_THROW((void)static_cast<int>(p), ov::Exception);
    ASSERT_THROW((void)p.as<int>(), ov::Exception);
}

TEST_F(AnyTests, AnyAsInts) {
    std::vector<int> ref = {1, 2, 3, 4, 5};
    Any p = ref;
    ASSERT_TRUE(p.is<std::vector<int>>());
    std::vector<int> test = p;
    ASSERT_EQ(ref.size(), test.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i], test[i]);
    }
}

TEST_F(AnyTests, AnyAsMapOfAnys) {
    std::map<std::string, Any> refMap;
    refMap["testParamInt"] = 4;
    refMap["testParamString"] = "test";
    Any p = refMap;
    bool isMap = p.is<std::map<std::string, Any>>();
    ASSERT_TRUE(isMap);
    std::map<std::string, Any> testMap = p;

    ASSERT_NE(testMap.find("testParamInt"), testMap.end());
    ASSERT_NE(testMap.find("testParamString"), testMap.end());

    int testInt = testMap["testParamInt"];
    std::string testString = testMap["testParamString"];

    ASSERT_EQ(refMap["testParamInt"].as<int>(), testInt);
    ASSERT_EQ(refMap["testParamString"].as<std::string>(), testString);
}

TEST_F(AnyTests, AnyNotEmpty) {
    Any p = 4;
    ASSERT_FALSE(p.empty());
}

TEST_F(AnyTests, AnyEmpty) {
    Any p;
    ASSERT_TRUE(p.empty());
}

TEST_F(AnyTests, AnyClear) {
    Any p = 4;
    ASSERT_FALSE(p.empty());
    p = {};
    ASSERT_TRUE(p.empty());
}

TEST_F(AnyTests, AnysNotEqualByType) {
    Any p1 = 4;
    Any p2 = "string";
    ASSERT_TRUE(p1 != p2);
    ASSERT_FALSE(p1 == p2);
}

TEST_F(AnyTests, AnysNotEqualByValue) {
    Any p1 = 4;
    Any p2 = 5;
    ASSERT_TRUE(p1 != p2);
    ASSERT_FALSE(p1 == p2);
}

TEST_F(AnyTests, AnysEqual) {
    Any p1 = 4;
    Any p2 = 4;
    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
}

TEST_F(AnyTests, AnysStringEqual) {
    std::string s1 = "abc";
    std::string s2 = std::string("a") + "bc";
    Any p1 = s1;
    Any p2 = s2;
    ASSERT_TRUE(s1 == s2);
    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
}

TEST_F(AnyTests, MapOfAnysEqual) {
    std::map<std::string, Any> map0;
    map0["testParamInt"] = 4;
    map0["testParamString"] = "test";
    const auto map1 = map0;

    Any p0 = map0;
    Any p1 = map1;
    ASSERT_TRUE(p0 == p1);
    ASSERT_FALSE(p0 != p1);
}

TEST_F(AnyTests, CompareAnysWithoutEqualOperator) {
    class TestClass {
    public:
        TestClass(int test, int* testPtr) : test(test), testPtr(testPtr) {}

    private:
        int test;
        int* testPtr;
    };

    TestClass a(2, reinterpret_cast<int*>(0x234));
    TestClass b(2, reinterpret_cast<int*>(0x234));
    TestClass c(3, reinterpret_cast<int*>(0x234));
    Any parA = a;
    Any parB = b;
    Any parC = c;

    ASSERT_THROW((void)(parA == parB), ov::Exception);
    ASSERT_THROW((void)(parA != parB), ov::Exception);
    ASSERT_THROW((void)(parA == parC), ov::Exception);
    ASSERT_THROW((void)(parA != parC), ov::Exception);
}

TEST_F(AnyTests, AnyRemovedRealObject) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        DestructorTest t;
        Any p1 = t;
    }
    ASSERT_EQ(2, DestructorTest::constructorCount);
    ASSERT_EQ(2, DestructorTest::destructorCount);
}

TEST_F(AnyTests, AnyRemovedRealObjectWithDuplication) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        DestructorTest t;
        Any p = t;
        ASSERT_EQ(0, DestructorTest::destructorCount);
        p = t;
        ASSERT_EQ(1, DestructorTest::destructorCount);
    }
    ASSERT_EQ(3, DestructorTest::constructorCount);
    ASSERT_EQ(3, DestructorTest::destructorCount);
}

TEST_F(AnyTests, AnyRemovedRealObjectPointerWithDuplication) {
    ASSERT_EQ(0, DestructorTest::constructorCount);
    ASSERT_EQ(0, DestructorTest::destructorCount);
    {
        auto* t = new DestructorTest();
        Any p = t;
        ASSERT_EQ(1, DestructorTest::constructorCount);
        ASSERT_EQ(0, DestructorTest::destructorCount);
        p = t;
        ASSERT_TRUE(p.is<DestructorTest*>());
        DestructorTest* t2 = p;
        ASSERT_EQ(0, DestructorTest::destructorCount);
        delete t;
        auto* t3 = p.as<DestructorTest*>();
        ASSERT_EQ(t2, t3);
    }
    ASSERT_EQ(1, DestructorTest::constructorCount);
    ASSERT_EQ(1, DestructorTest::destructorCount);
}

void PrintTo(const Any& object, std::ostream* stream) {
    if (object.empty() || !stream) {
        return;
    }
    object.print(*stream);
}

TEST_F(AnyTests, PrintToEmptyAnyDoesNothing) {
    Any p;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToIntAny) {
    int value = -5;
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, PrintToUIntAny) {
    unsigned int value = 5;
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, PrintToSize_tAny) {
    std::size_t value = 5;
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::to_string(value));
}

TEST_F(AnyTests, PrintToFloatAny) {
    Any p = 5.5f;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{"5.5"});
}

TEST_F(AnyTests, PrintToStringAny) {
    std::string value = "some text";
    Any p = value;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), value);
}

TEST_F(AnyTests, PrintToVectorOfIntsAnyDoesNothing) {
    Any p = std::vector<int>{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToVectorOfUIntsAnyDoesNothing) {
    Any p = std::vector<unsigned int>{0, 1, 2, 3, 4, 5};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToVectorOfSize_tAnyDoesNothing) {
    Any p = std::vector<std::size_t>{0, 1, 2, 3, 4, 5};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToVectorOfFloatsAnyDoesNothing) {
    Any p = std::vector<float>{0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToVectorOfStringsAnyDoesNothing) {
    Any p = std::vector<std::string>{"zero", "one", "two", "three", "four", "five"};
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, PrintToMapOfAnysDoesNothing) {
    std::map<std::string, Any> refMap;
    refMap["testParamInt"] = 4;
    refMap["testParamString"] = "test";
    Any p = refMap;
    std::stringstream stream;
    ASSERT_NO_THROW(p.print(stream));
    ASSERT_EQ(stream.str(), std::string{});
}

TEST_F(AnyTests, constructFromVariantImpl) {
    auto parameter = Any{4};
    auto get_impl = [&] {
        return std::make_shared<VariantImpl<int>>();
    };
    auto other_parameter = Any{get_impl()};
}

TEST_F(AnyTests, dynamicPointerCastToVariant) {
    Any p = std::make_shared<VariantWrapper<std::string>>("42");
    auto str_variant = std::dynamic_pointer_cast<VariantWrapper<std::string>>(p);
    ASSERT_EQ("42", str_variant->get());
}

TEST_F(AnyTests, asTypePtrToVariant) {
    Any p = std::make_shared<VariantWrapper<std::string>>("42");
    auto str_variant = ov::as_type_ptr<VariantWrapper<std::string>>(p);
    ASSERT_EQ("42", str_variant->get());
}

TEST_F(AnyTests, castToVariant) {
    {
        Any p = std::make_shared<VariantWrapper<std::string>>("42");
        std::shared_ptr<VariantWrapper<std::string>> str_variant = p;
        ASSERT_EQ("42", str_variant->get());
    }
    {
        Any p = std::make_shared<VariantWrapper<std::string>>("42");
        auto f = [](const std::shared_ptr<VariantWrapper<std::string>>& str_variant) {
            ASSERT_NE(nullptr, str_variant);
            ASSERT_EQ("42", str_variant->get());
        };
        f(p);
    }
    {
        Any p = std::make_shared<VariantWrapper<std::string>>("42");
        auto f = [](std::shared_ptr<VariantWrapper<std::string>>& str_variant) {
            ASSERT_NE(nullptr, str_variant);
            ASSERT_EQ("42", str_variant->get());
        };
        f(p);
    }
    {
        std::shared_ptr<Variant> v = std::make_shared<VariantWrapper<std::string>>("42");
        Any p = v;
        auto f = [](std::shared_ptr<VariantWrapper<std::string>>& str_variant) {
            ASSERT_NE(nullptr, str_variant);
            ASSERT_EQ("42", str_variant->get());
        };
        f(p);
    }
}
