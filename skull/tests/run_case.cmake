# Fuehrt ein .skull-Skript aus und prueft Exit-Code und Ausgabe.
# Aufruf (aus CTest):
#   cmake -DSKULL=<exe> -DSCRIPT=<datei.skull> -DWORKDIR=<dir>
#         -DEXPECT_EXIT=<code> [-DEXPECT_MATCH=<regex>] [-DEXPECT_NO_MATCH=<regex>]
#         -P run_case.cmake
#
# Das Skript laeuft in einem eigenen Skript statt als einfacher add_test, weil
# CTest bei PASS_REGULAR_EXPRESSION den Exit-Code ignoriert. Hier muss beides
# stimmen. Es braucht weder Python noch bash und laeuft deshalb auch unter Windows.

cmake_minimum_required(VERSION 3.15)

foreach(var SKULL SCRIPT WORKDIR EXPECT_EXIT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "run_case.cmake: ${var} fehlt")
    endif()
endforeach()

execute_process(
    COMMAND "${SKULL}" "${SCRIPT}"
    WORKING_DIRECTORY "${WORKDIR}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE  err
    TIMEOUT 600)

set(all "${out}\n${err}")

if(NOT "${rc}" STREQUAL "${EXPECT_EXIT}")
    message(FATAL_ERROR
        "${SCRIPT}: Exit-Code '${rc}', erwartet '${EXPECT_EXIT}'\n"
        "---- Ausgabe ----\n${all}")
endif()

if(DEFINED EXPECT_MATCH AND NOT "${EXPECT_MATCH}" STREQUAL "")
    if(NOT all MATCHES "${EXPECT_MATCH}")
        message(FATAL_ERROR
            "${SCRIPT}: Ausgabe passt nicht auf '${EXPECT_MATCH}'\n"
            "---- Ausgabe ----\n${all}")
    endif()
endif()

if(DEFINED EXPECT_NO_MATCH AND NOT "${EXPECT_NO_MATCH}" STREQUAL "")
    if(all MATCHES "${EXPECT_NO_MATCH}")
        message(FATAL_ERROR
            "${SCRIPT}: Ausgabe enthaelt unerwartet '${EXPECT_NO_MATCH}'\n"
            "---- Ausgabe ----\n${all}")
    endif()
endif()
