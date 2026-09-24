# The monitor page (monitor_page.html) is compiled into the server: it is split into raw string
# literals of at most 8000 bytes (MSVC limits one literal to ~16 KiB) and rewritten only when
# the page changes, so an edit re-runs configure and rebuilds one object. The page is ASCII.
set(_monitor_page "${CMAKE_CURRENT_SOURCE_DIR}/monitor_page.html")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_monitor_page}")
file(READ "${_monitor_page}" _monitor_text)
string(LENGTH "${_monitor_text}" _monitor_length)
set(_monitor_literals "")
set(_monitor_offset 0)
while(_monitor_offset LESS _monitor_length)
  string(SUBSTRING "${_monitor_text}" ${_monitor_offset} 8000 _monitor_piece)
  string(APPEND _monitor_literals "    R\"NINFER_PAGE(${_monitor_piece})NINFER_PAGE\"\n")
  math(EXPR _monitor_offset "${_monitor_offset} + 8000")
endwhile()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/monitor_page.cpp.in"
  "// Generated from src/serve/monitor_page.html by src/serve/CMakeLists.txt.\n"
  "#include \"serve/monitor_page.h\"\n\n"
  "namespace ninfer::serve {\n\n"
  "std::string_view monitor_page() noexcept {\n"
  "    static constexpr char kPage[] =\n${_monitor_literals};\n"
  "    return {kPage, sizeof(kPage) - 1};\n"
  "}\n\n"
  "} // namespace ninfer::serve\n")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/monitor_page.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/monitor_page.cpp" COPYONLY)

add_library(ninfer_serve STATIC
  http_transport.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  operational_log.cpp
  request_log.cpp
  http_server.cpp
  openai_responses_http.cpp
  serve_metrics.cpp
  serve_options.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/monitor_page.cpp"
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_product_logging ${NINFER_CUDART_TARGET})
