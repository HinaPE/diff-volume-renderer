#pragma once

#include <string>

#include "hotpath/hp.h"

namespace dvren {

enum class StatusCode {
    kOk = 0,
    kInvalidArgument,
    kOutOfMemory,
    kNotImplemented,
    kUnsupported,
    kInternalError
};

class Status {
public:
    Status() = default;
    Status(StatusCode code, std::string message);

    static Status Ok();
    static Status FromHotpath(hp_status code, std::string message = {});

    [[nodiscard]] bool ok() const;
    [[nodiscard]] explicit operator bool() const { return ok(); }
    [[nodiscard]] StatusCode code() const;
    [[nodiscard]] const std::string& message() const;
    [[nodiscard]] std::string ToString() const;

private:
    StatusCode code_{StatusCode::kOk};
    std::string message_{};
};

}  // namespace dvren

