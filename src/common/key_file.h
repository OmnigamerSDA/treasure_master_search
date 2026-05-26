#ifndef KEY_FILE_H
#define KEY_FILE_H

#include "data_sizes.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace key_file
{

inline std::string trim(const std::string& s)
{
    std::size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) begin++;
    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(begin, end - begin);
}

inline std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
    {
        fields.push_back(trim(field));
    }
    return fields;
}

inline bool parse_uint32(const std::string& s, uint32& out)
{
    const std::string t = trim(s);
    if (t.empty()) return false;
    try
    {
        std::size_t pos = 0;
        const unsigned long v = std::stoul(t, &pos, 0);
        if (pos == 0 || v > 0xFFFFFFFFul) return false;
        out = static_cast<uint32>(v);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

inline bool looks_like_key_column(const std::string& name)
{
    const std::string n = lower(trim(name));
    return n == "key" || n == "key_hex" || n == "fixed_key" ||
           n == "fixed_key_hex" || n == "key_dec";
}

inline int find_key_column(const std::vector<std::string>& header)
{
    for (std::size_t i = 0; i < header.size(); i++)
    {
        if (looks_like_key_column(header[i])) return static_cast<int>(i);
    }
    return -1;
}

inline bool parse_key_from_fields(const std::vector<std::string>& fields, int key_column, uint32& out)
{
    if (key_column >= 0)
    {
        if (static_cast<std::size_t>(key_column) >= fields.size()) return false;
        return parse_uint32(fields[static_cast<std::size_t>(key_column)], out);
    }

    for (const std::string& field : fields)
    {
        const std::string t = trim(field);
        if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X'))
        {
            return parse_uint32(t, out);
        }
    }

    if (!fields.empty()) return parse_uint32(fields[0], out);
    return false;
}

inline std::vector<uint32> read_keys(const std::string& path, std::size_t limit)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open key list: " + path);

    std::vector<uint32> keys;
    std::string line;
    int key_column = -1;
    bool header_checked = false;

    while (std::getline(f, line))
    {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        line = trim(line);
        if (line.empty()) continue;

        const std::vector<std::string> fields = split_csv_line(line);
        if (!header_checked)
        {
            header_checked = true;
            key_column = find_key_column(fields);
            if (key_column >= 0)
            {
                uint32 ignored = 0;
                if (!parse_key_from_fields(fields, key_column, ignored)) continue;
            }
        }

        uint32 key = 0;
        if (parse_key_from_fields(fields, key_column, key))
        {
            keys.push_back(key);
            if (limit > 0 && keys.size() >= limit) break;
        }
        else
        {
            std::cerr << "skipped key row: " << line << "\n";
        }
    }

    return keys;
}

} // namespace key_file

#endif // KEY_FILE_H
