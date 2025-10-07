#include "dvren/core/status.hpp"

#include <utility>

namespace dvren {

namespace {

StatusCode MapHotpathStatus(hp_status code) {
    switch (code) {
        case HP_STATUS_SUCCESS:
            return StatusCode::kOk;
        case HP_STATUS_INVALID_ARGUMENT:
            return StatusCode::kInvalidArgument;
        case HP_STATUS_OUT_OF_MEMORY:
            return StatusCode::kOutOfMemory;
        case HP_STATUS_NOT_IMPLEMENTED:
            return StatusCode::kNotImplemented;
        case HP_STATUS_UNSUPPORTED:
            return StatusCode::kUnsupported;
        case HP_STATUS_INTERNAL_ERROR:
        default:
            return StatusCode::kInternalError;
    }
}

const char* StatusCodeToString(StatusCode code) {
    switch (code) {
        case StatusCode::kOk:
            return "ok";
        case StatusCode::kInvalidArgument:
            return "invalid_argument";
        case StatusCode::kOutOfMemory:
            return "out_of_memory";
        case StatusCode::kNotImplemented:
            return "not_implemented";
        case StatusCode::kUnsupported:
            return "unsupported";
        case StatusCode::kInternalError:
        default:
            return "internal_error";
    }
}

}  // namespace

Status::Status(StatusCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

Status Status::Ok() {
    return Status{};
}

Status Status::FromHotpath(hp_status code, std::string message) {
    return Status(MapHotpathStatus(code), std::move(message));
}

bool Status::ok() const {
    return code_ == StatusCode::kOk;
}

StatusCode Status::code() const {
    return code_;
}

const std::string& Status::message() const {
    return message_;
}

std::string Status::ToString() const {
    if (ok()) {
        return "ok";
    }
    if (message_.empty()) {
        return std::string(StatusCodeToString(code_));
    }
    return std::string(StatusCodeToString(code_)) + ": " + message_;
}

}  // namespace dvren

