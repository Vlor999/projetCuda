// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM
#pragma once

#include <cstdint>
#include <limits>
#include "Random.hpp"

struct TestType
{
  int x;
  int y;
  int z;
  int w;

  bool operator==(const TestType& other) const {
    return x == other.x && y == other.y && z == other.z && w == other.w;
  }

  bool operator!=(const TestType& other) const {
    return !(*this == other);
  }

  void initRandom()
  {
    x = Random::GetInt(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    y = Random::GetInt(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    z = Random::GetInt(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    w = Random::GetInt(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
  }
};

__host__ __device__
inline bool test_hash0(const TestType& t, uint32_t index)
{
    uint32_t value = t.x;
    return ((value * 1920351) % 3) == 2;
}

__host__ __device__
inline bool test_hash1(const TestType& t, uint32_t index)
{
    return ((index * 1920351) % 7) != 1;
}

__host__ __device__
inline bool test_hash2(const TestType& t, uint32_t index)
{
    return ((index * 1920351) % 7) == 3;
}

namespace std {
  template<> struct hash<TestType> {
    size_t operator()(const TestType& obj) const
    {
      auto h = hash<int>()(obj.x);
      h ^= hash<int>()(obj.y);
      h ^= hash<int>()(obj.z);
      h ^= hash<int>()(obj.w);
      return h;
    }
  };
}

// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM
