#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool
{
private:

    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::condition_variable cv_finished_;
    bool stop_;
    size_t unfinished_tasks_;

public:

    ThreadPool(size_t thread_count = std::max(std::thread::hardware_concurrency(), 1u))
        : stop_(false)
        , unfinished_tasks_(0)
    {
        auto worker = [this]()
        {
            while (true)
            {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);

                    // wait until a stop is requested or until there's a task to complete
                    this->cv_.wait(lock, [this]
                    {
                        return this->stop_ || !this->tasks_.empty();
                    });

                    // if stopping and there are no tasks to complete, exit the loop
                    if (this->stop_ && this->tasks_.empty())
                    {
                        return;
                    }

                    // retrieve the next task from the queue
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }

                // execute the task
                task();

                // upon task completion, update count of unfinished tasks
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    --unfinished_tasks_;
                    if (unfinished_tasks_ == 0)
                    {
                        cv_finished_.notify_all();
                    }
                }
            }
        };

        // start the worker threads
        for (size_t i = 0; i < thread_count; ++i)
        {
            threads_.emplace_back(worker);
        }
    }

    auto Threads() const
    {
        return threads_.size();
    }

    template <typename Function, typename... Arguments>
    void Enqueue(Function&& function, Arguments&&... args)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            if (stop_)
            {
                throw std::runtime_error("cannot enqueue on a stopped thread pool");
            }

            tasks_.emplace(std::bind(std::forward<Function>(function), std::forward<Arguments>(args)...));
            ++unfinished_tasks_;
        }
        cv_.notify_one();
    }

    void Wait()
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_finished_.wait(lock, [this]()
        {
            return unfinished_tasks_ == 0;
        });
    }

    ~ThreadPool()
    {
        Shutdown();
    }

    void Shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();

        for (auto& thread : threads_)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
    }
};
