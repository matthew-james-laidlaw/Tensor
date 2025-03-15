#include <gtest/gtest.h>

#include <Internal/ThreadPool.hpp>

#include <atomic>
#include <chrono>
#include <thread>

TEST(ThreadPoolTests, ConstructorMaximumThreadCount)
{
    ThreadPool pool;
    EXPECT_EQ(pool.Threads(), std::thread::hardware_concurrency());
}

TEST(ThreadPoolTests, ConstructorProvidedThreadCount)
{
    ThreadPool pool(4);
    EXPECT_EQ(pool.Threads(), 4);
}

TEST(ThreadPoolTests, SingleTaskIncrementsCounter)
{
    ThreadPool pool;
    
    std::atomic<int> counter = 0;

    pool.Enqueue([&counter]()
    {
        counter.fetch_add(1);
    });

    pool.Wait();

    EXPECT_EQ(counter.load(), 1);
}

TEST(ThreadPoolTests, MultipleTasksIncrementCounter)
{
    ThreadPool pool;
    
    std::atomic<int> counter = 0;

    for (size_t i = 0; i < 100; ++i)
    {
        pool.Enqueue([&counter]()
        {
            counter.fetch_add(1);
        });
    }

    pool.Wait();

    EXPECT_EQ(counter.load(), 100);
}
