#include <Common/FileUtil.hpp>

namespace mandel {
namespace fs {
std::string FileUtil::join(const std::string basePath, const std::string additional) {
  tPath path_basePath = generic_fs::absolute(tPath(basePath));
  return (path_basePath / tPath(additional)).string();
}

std::string FileUtil::absPath(const std::string path) { return generic_fs::absolute(tPath(path)).string(); }

std::string FileUtil::dirPath(const std::string path) { return generic_fs::absolute(tPath(path)).parent_path().string(); }

std::string FileUtil::baseName(const std::string path) { return generic_fs::absolute(tPath(path)).filename().string(); }

std::string FileUtil::extension(const std::string path) { return tPath(path).extension().string(); }

std::string FileUtil::cwd() { return generic_fs::current_path().string(); }

void FileUtil::mkdirs(const std::string path) {
  tPath path_basePath = generic_fs::absolute(tPath(path));
  generic_fs::create_directories(path_basePath);
}

bool FileUtil::exists(const std::string path) { return generic_fs::exists(generic_fs::absolute(tPath(path))); }

bool FileUtil::isAbsolute(const std::string path) { return tPath(path).is_absolute(); }

}  // namespace fs
}  // namespace mandel