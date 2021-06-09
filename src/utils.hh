#pragma once

#include <cassert>
#include <iostream>

class Log
{
public:
  template <typename... Targs>
  static inline void err(Targs... args)
  {
    std::cerr << "\033[1;31m[ERROR]\033[0m ";
    print_pack(std::cerr, args...);
    std::cerr << '\n';
  }

  template <typename... Targs>
  static inline void warn(Targs... args)
  {
    std::cerr << "\033[1;33m[WARN]\033[0m ";
    print_pack(std::cerr, args...);
    std::cerr << '\n';
  }

  template <typename... Targs>
  static inline void info(Targs... args)
  {
    std::cerr << "\033[1;32m[INFO]\033[0m ";
    print_pack(std::cerr, args...);
    std::cerr << '\n';
  }

  template <typename... Targs>
  static inline void dbg([[maybe_unused]] Targs... args)
  {
#ifdef DEBUG
    std::cerr << "\033[1;34m[DEBUG]\033[0m ";
    print_pack(std::cerr, args...);
    std::cerr << '\n';
#endif
  } 

private:

  template <typename T>
  static void print_pack(std::ostream& os, T value)
  {
    os << value;
  }

  template <typename T, typename... Targs>
  static void print_pack(std::ostream& os, T value, Targs... args)
  {
    os << value;

    if (sizeof...(Targs) != 0)
      print_pack(os, args...);
  }
};

#if defined DEBUG
  #define ASSERT(expr) assert(expr)
#else
  #define ASSERT(expr)
#endif

#if defined __clang__ || defined __GNUC__
  #define LIKELY(expr) __builtin_expect(!!(expr), 1)
  #define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#else
  #define LIKELY(expr) (expr)
  #define UNLIKELY(expr) (expr)
#endif

#if defined __clang__ || defined __GNUC__
  #define UNREACHABLE_BUILTIN() __builtin_unreachable()
#else
  #define UNREACHABLE_BUILTIN()
#endif

#define UNREACHABLE() do {                         \
    ASSERT(!("Unreachable code reached ?!"));     \
    UNREACHABLE_BUILTIN();                        \
   } while (0)

#define NOT_IMPLEMENTED_YET() throw std::runtime_error("Not implemented yet !");

#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
  #define GCC
#elif defined(__llvm__)
  #define CLANG
#endif

/*
 * Print a pair<T1, T2> on os.
 */
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& p);

/*
 * Check if it is a collection.
 */
template <typename T, typename _ = void>
struct is_collection : std::false_type {};

template<typename... Ts>
struct is_collection_helper {};

template<typename T>
struct is_collection<
  T,
  std::conditional_t<
    false,
    is_collection_helper<
      typename T::value_type,
      typename T::size_type,
      typename T::allocator_type,
      typename T::iterator,
      typename T::const_iterator,
      decltype(std::declval<T>().size()),
      decltype(std::declval<T>().begin()),
      decltype(std::declval<T>().end()),
      decltype(std::declval<T>().cbegin()),
      decltype(std::declval<T>().cend())
    >,
    void
  > 
> : public std::true_type{};

/*
 * Generic print for collection.
 */
template <
  typename Collection,
  std::enable_if_t<
      is_collection<Collection>::value
      & !std::is_same<Collection, std::string>::value
      & !std::is_array<Collection>::value
  >* = nullptr
>
std::ostream& operator<<(std::ostream& os, const Collection& col)
{
  os << "[ ";
  for (const auto& val : col)
    os << val << ' ';
  return os << ']';
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& p)
{
  return os << '(' << p.first << ", " << p.second << ')';
}
