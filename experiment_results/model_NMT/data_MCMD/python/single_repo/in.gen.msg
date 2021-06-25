Issue : An exception in Tkinter callback no longer crashes the program
Update Tarek ( ) entry .
closes bpo - 39261 : Remove dead API from pyinit_config . ( GH - 17907 )
Issue : Skip test_case_operation_overflow on computers with low memory .
Capitalize & use periods for \\ seemodule explanation parameter .
Shut up an occaisonal buildbot error due to test files being left around .
Merged revisions 76774 via svnmerge from
bpo - 40221 : Fix multiprocessing PidfdChildWatcher on waitpid ( ) error ( GH - 17477 )
Merged revisions 61034 - 61036 , 61038 - 61048 via svnmerge from
: fix missing paste error message .
merge heads
Merged revisions 82769 via svnmerge from
Use PyModule_AddIntConstant ( ) instead of creating a private helper function .
bpo - 30566 : Fix test_flush_return_value on FreeBSD ( GH - 9451 )
Fix two bug in Doc / make the text on Windows .
M . - A . Lemburg < mal @ lemburg . com > :
Add a note about the PSA , since the fate of the docs
Merged revisions 81838 via svnmerge from
Issue : Fixed Fraction and update of True / False .
Merged revisions 72079 via svnmerge from
promote some shifts to unsigned , so as not to invoke undefined behavior
Issue : Fix a crash in the TextIOWrapper . tell method on 64 - bit platforms .
Rename " Save " type from Makefile .
Issue : Factor out the _SuppressCoreFiles context manager into test . support .
Issue : The trace module now uses Argument Clinic .
Optimize the freeblock ( ) function .
Issue : Fix typo in socket doc ; Patch by anatoly techtonik
Add Samuele Pedroni
: fix typo .
asyncio : sync with Tulip
merge 3 . 5 ( )
SRE C API , bpo - 31370 : ( )
Various edits updates .
Issue : FileIO . extractfile ( ) failed to resolve symbolic links when
bpo - 40528 : Fix IDLE ProactorLoopCtrlC ( )
Added Raymond Hettinger , CPA .
merge with 3 . 2
Patch for issue : don', "'", 't depend on Cygwin , as chmod doesn', "'", 't work reliably there
bpo - 29854 : Fix pymain_init_pycache_prefix ( ) ( )
Issue : Fix a crash when accessing the ` ` cell_contents ` `
Added version info for name2codepoint and codepoint2name
Add optional to list of ( ) function .
patch : don', "'", 't define _XOPEN_SOURCE on Mac OS Panther
Patches describing the statvfs ( ) and fstatvfs ( ) functions . Additional
bpo - 34519 : Add Python 3 . 7 buildbot link to', "'", '. github / CONTRIBUTING . rst', "'", '( GH - 7532 )
Issue : An exception in Tkinter callback no longer crashes the program
GetoptError is always initialized with exactly two parameters , so simplify
merge 3 . 4
bpo - 30636 : Fix the code for the help ( )
when print ( ) gets unicode arguments , sep and end should be unicode by default
Merged revisions 80610 via svnmerge from
Oops , PLstring funcs should be " pascal " . Funny this never caused problems before . . .
: fix typo in kqueue , occurring in certain error messages .
Make sure the root windows gets destroyed
Move building nuget packages .
Put all thrashing instead of USE_UCS4_STORAGE and Py_UNICODE_SIZE
Issue : Fix a comment in the TextIOWrapper . encode method on 64 - bit platforms .
Issue : Fix a crash in the TextIOWrapper . tell method on 64 - bit platforms .
: make linecache docs reflect that all files are treated the same .
Various means !
Merged revisions 71280 via svnmerge from
: Preserve permissions of mbox , MMDF and Babyl mailbox files on flush ( )
Merged revisions 81838 via svnmerge from
Issue : Add SHA - 3 and SHAKE support to hashlib module .
AddrlistClass . getdomainliteral ( ) : rfc822 requires that the domain
typo .
Remove __repr__ that doesn', "'", 't have any advantage compared to the standard one .
Merged revisions 86985 via svnmerge from
Fix pathext fix for issue 7852 .
This fixes was manufactured by Christian Tismer .
Add zlib
Issue : multiprocessing : Remove some unused assignments to ob_size in compatible
explicitly cast away constness to silence compiler warning
Issue : Remove unneeded words about RuntimeError
: Fix typo in multiprocessing tests .
Merged revisions 71540 , 78308 via svnmerge from
Make sure the AST is always capitalized .
Fix typo .
Update the __version__
Issue : Merge with 3 . 5
Make check_warnings error messages more informative
Patch 1352 ( continued in issue 1329 ) by Christian Heimes .
Added PyExc_Exception and PyErr_NewException .
Issue : Fix a test_signal failure : A delivery order is only defined for
Use math . py and math . fsum
Merged revisions 79866 - 79867 via svnmerge from
Added contributor entry .
Branch merge
In isinstance
Merge with 3 . 3
Minor edits
Remove all comments on Mac
pstats - up analysis module .
Issue : Document NASM 2 . 10 + as requirement for building OpenSSL 1 . 0 . 1 on Windows
Mark Hammond : Added dependency of winsound project on python16
Merged revisions 70397 via svnmerge from
Issue : set os . path . join support :
: fix typo .
Issue : Fix test_gdb failures on Windows in Ubuntu . Patch by David Edelsohn .
bunch boolean example for script_from_examples ( ) .
Prep for Python 3 . 1 !
Merged revisions 78322 via svnmerge from
Issue : Add SIGWINCH handler for Readline 6 . 3 + support , by Eric Price
- Issue : Add a new PyFrame_FastToLocalsWithError ( ) function to handle
Added __members__ to FInfo object .
Initial revision
Add example for PyErr_SetFromWindowsErr ( ) docs .
Fix typo in functional module
Merge with 3 . 2
: fix documentation for the email docs .
Moshe Zadka < mzadka @ geocities . com > :
Issue : Fix validation of relative imports in parser module .
whoops , I wanted that commented out by default , will add doc to Misc
Issue : Fix ProactorEventLoop . close ( ) on Windows for files larger than 2 GB
Last CW Pro 5 . 3 projects .
Patch by kragen @ pobox . com : When tracing is turned on , lines shorter
Script to pybench 9 . main ( sigh ) .
Documentation for the overview module .
Issue : Fix occasional testing error message
Fix issue11082 - Reject str for POST data with a TypeError . Document the need to explicitly encode to bytes when using urlencode .
Fix typo
What', "'", 's New in Python 3 . 2 : update Unicode section
Added " Internet manager " to unpickling index . py .
* wdbframewin . py : Added MAX_MAXCHAR ( ) failure ( was
Fix a typo in the distutils module .
RFE : Socket objects', "'", 'family , type and proto properties are
Merge heads
Issue : Check for ERROR_ACCESS_DENIED in Windows implementation of os . stat ( )
Removed the', "'", 'ensure_', "'", 'command .
Merge .
merge with 3 . 4
cleaned up distutils tests and ack .
Add items
Add test for the new - comparison option . ( oops )
whatsnew : filecmp . DEFAULT_IGNORES ( )
Issue : Corrected fileConfig ( ) documentation .
merge heads
Fix typo in NEWS leak .
err is no longer used
Merge 3 . 5 ( issue )
Add a test for the HeaderParser class .
use Py_UNICODE_WIDE instead of USE_UCS4_STORAGE and Py_UNICODE_SIZE
ests by Sjoerd
Closes : Doc / Improved doc .
Merged revisions 68302 via svnmerge from
Another _winreg hack to Sjoerd .
Merge with 3 . 4
Use Tcl_SetObjResult instead of Tcl_SetResult .
A number of small adjustments .
condense swap changes :
Issue : Fix a leak in the TextIOWrapper . tell method on 64 - bit platforms .
Merged revisions 75143 via svnmerge from
Add whatsnew for new section
Merge
Added a reminder
Issue : TarFile . extractfile ( ) failed to resolve symbolic links when
bpo - 35808 : Retire pgen and use pgen2 to generate the parser ( GH - 11814 )

Issue : Restore OSError constructor .
Issue : Merge from 3 . 5
Added __members__ to FInfo object .
Add missing decref
Merged revisions 81820 via svnmerge from
Merged revisions 82806 via svnmerge from
Issue : Fix a crash in the TextIOWrapper . tell method on 64 - bit platforms .
NEWS for r60121
Merge Py Idle changes :
Issue : Generator . new_module ( ) uses Unicode
Fix test_fnmatch .
- gusiprefs . h added ( note : must be in application / applet , not in
asyncio : sync with Tulip
Marc - Andre Lemburg < mal @ lemburg . com > :
: a better test
bpo - 34019 : Fix section in the ast module . ( )
Merged revisions 73931 via svnmerge from
fix indentation
Add example
target libinstall :
Add entry for curses . panel documentation .
Issue : Fix ProactorEventLoop . close ( )
: fix _dummy_thread import example .
Adding newind . py happy .
Add pgrep ( ) function , which is like grep / egrep / emgrep but uses Perl
Merged revisions 74675 via svnmerge from
this is better on windows
testing control filename in the absence of snprintf on Windows .
Issue : Keep url ( ) check in unicode_compare ( ) .
Delete news entry .
Issue : Avoid WSGIRequestHandler doing the fork bomb when using multiprocessing
Added tag v3 . 2 . 1rc2 for changeset 5df549718fb4
Add missing decref
Merged revisions 59407 - 59422 via svnmerge from
Fix refleak in the error case
Turn test_nis in stead of mremap ( )
Add bigmemtest tests for bool
Add news item for 57f0af61da53 .
Issue : The modulefinder module now supports extended opcode arguments .
Added tag v3 . 2 . 1rc2 for changeset 5df549718fb4
closes bpo - 34868 : Fix constness of members of tok_state struct . ( GH - 18600 )
Link to list of keywords in the laguage reference ( GH - 18024 )
- general markup cleanup
Merge with 3 . 4
: fix docstring .
Fix waiter cancellation in xxlimited . c
: fix typo in init doc ; Patch by anatoly techtonik
Add Jason Tishler to the ACKS file
Issue : multiprocessing now opens the setup script in binary mode to read the
use macros
bpo - 35214 : Initial clang MemorySanitizer support ( GH - 10479 )
Added comments about pimp and bundlebuilder to the Mac section .
The first suport for augmented assignment .
merge with 3 . 2
Merge heads .
Issue : Remove uses of assignments .
[ Bug ] Minor Mac nits
Backout changeset 46393019b650
Merged revisions 66539 via svnmerge from
Issue : Add SHA ( ) function to BaseEventLoopTests module
Make decimal . ContextManager a private implementation detail of decimal . localcontext ( )
Fix typo .
Use correct names .
Fix sdist failure on OS X .
Issue : logging logging docs
Whitespace normalization .
Fix - Wstrict - prototypes warning in thread_pthread . h . ( GH - 21477 )
Issue : The asynchat . async_chat . handle_read ( ) method now ignores
issue8053 - logic was inverted on which platforms .
Merge heads
labs ( ) takes a long for an input .
Minor documentation tweaks and simpler update ( ) example .
Merge 3 . 4 ( asyncio doc )
Issue : Make sure that Python - ast . h is properly taken into account in the
Issue : The modulefinder module now supports extended opcode arguments .
Typo fix .
Merge 3 . 2
bpo - 33383 : Improve crash in call_readline ( ) ( GH - 728 )
Merged revisions 66383 via svnmerge from
Fix issue : An errornous __length_hint__ can make list ( ) raise a
bpo - 38858 : Add _PyPathConfig_ComputeArgv0 ( ) ( )
Issue : Use test_both ( ) consistently in test_importlib .
Merged revisions 73824 , 78887 , 78895 , 78900 , 79024 via svnmerge from
Merged revisions 78322 via svnmerge from
bpo - 34043 : Fix test_asyncio ProactorLoopCtrlC ( GH - 14074 )
: add a missing suppress_crash_popup ( ) function to handle
Issue : Merge from 3 . 5
SF patch : Replace push / pop clusters in ceval . c
Issue : xxlimited . new_module ( ) now waits for the underlying thread state to be destroyed before returning .
Merged revisions 61207 via svnmerge from
Revert issue : fix indentation
Added docs for the spwd module .
Issue : Fix ElementTree . close ( ) on Windows for files larger than 2 GB
bpo - 29636 : Add PyFrame_GetBack ( ) function ( GH - 19765 )
Minor edits
PyBuffer_New ( ) : Apply error if size is negative ( the other
Merged revisions 82169 via svnmerge from
Use PyUnicode_FromFormat ( ) instead of Tcl_SetResult .
typo
New wishes
bpo - 32030 : _PyGILState_Init ( ) returns PyStatus ( GH - 18908 )
Issue : Improve cross - references in builtin types and functions documentation .
Added a way to say " use any resource except A " . For example , to run
merge heads
Fix incompatible warning .
Merged revisions 76366 via svnmerge from
In the multifile module .
Merged revisions 77794 via svnmerge from
Issue : Fix the urllib . request . Request . is_unverifiable deprecation documentation .
Issue : test_range with Tulip
Issue : Fix PyDict_New ( ) to handle correctly new_keys_object ( ) failure
asyncio : sync with Tulip
Merged revisions 80777 via svnmerge from
Issue : Enables bytes ( ) and resize_inplace ( ) fills also the Unicode strings
Rewrite a Py_IS_NAN call .
Closes : regex . flags has not only explicit flags but also those from the pattern
Issue : TarFile . extractfile ( ) failed to resolve symbolic links when
: fix typo in the sqlite module .
Add big entry for " To , " format
Issue : Merge from 3 . 5
Merged revisions 61687 - 60244 via svnmerge from
start numbering things 1 . 5b1 now
merge with 3 . 4
Fix a poorly worded error message
Merged revisions 84289 via svnmerge from
bpo - 15221 : Update os . path . is * ( ) documentation ( GH - 5185 )
Fix test_poll . py , thanks to Robert Bardos from docs @
: fix several factual errors .
: merge with 3 . 3 .
Issue : Fix call_function ( ) of ceval . c to handle PyTuple_New ( ) failure
Merge with 3 . 4
Added descrobject . c and mysnprintf . c
: fix typo in turtle . py .
Add a note about pow ( ) , which returns an intended
Add decl for PySuper_Type .
* AppleEvent round to infinite
Merge 3 . 5
Issue : Add tests for the spwd module . Original patch by Vajrasky Kok .
target libinstall', "'", 's new other API .
Merged revisions 75519 via svnmerge from
bpo - 36373 : Fix deprecation warnings in test_importlib . ( GH - 7144 )
: fix typo in turtle . py .
Added notes about __builtin__ . py .
: typos
Fix test_urllib2 failure on OS X .
eval_code2 ( ) : In the extended calling syntax opcodes , you must check
NEWS for issue
: merge with 3 . 3 .
Issue : Use specific asserts in http . cookiejar tests .
Fix typo .
Convert test_nis to compiler .
Issue : Merge with 3 . 4
merge 3 . 5
Merged revisions 67279 via svnmerge from
Assume nasm .
Merged revisions 83818 via svnmerge from
sys . check_interval = x - > sys . setcheckinterval ( x )
Make decimal . ContextManager a private implementation detail of decimal . localcontext ( )
se modern ( ) rather than compiler_obj .
bpo - 38558 : Link to further docs from walrus operator mention in tutorial ( GH - 16973 )
configure : fix HAVE_GETRANDOM_SYSCALL check
Added tag_prevrange analogous to rag_nextrange .
Fix bug .
Issue : Oops , the patch for codecs . encode / decode doc was written by
Issue : Remove outdated versionchanged directives .
one CERT_REQUIRED is enough
- PyType_Ready ( ) : Initialize the ob_type field to & PyType_Type if it', "'", 's
Issue : Document that encoding lines cannot follow non - comment lines .
Issue : Fix UnboundLocalError in AbstractDigestAuthHandler . get_algorithm_impls
st_dev / st_rdev should be unsigned long as dwVolumeSerialNumber , which it is set to , is a DWORD . This was fixed in and the overflow was mentioned in and seen by me on some machines .
Regenerated pydoc topics for Python 3 . 5 . 0rc1 .
Issue : Make plural form selection more lenient and accepting
Issue : Prevent conflict of UNICODE macros in cPickle .
Merged revisions 72060 via svnmerge from
Fix test_poll . py . By Christian Heimes .
Bump version to 3 . 2 . 2rc1 .
Fix incorrect stacklevel for DeprecationWarnings originating from the struct module .
Try to fix test_gdb on s390x buildbots
Close : Adjust callable ( ) warning : callable ( ) is only not supported in
Issue : Fix time . epoll . PathFinder to support
Remove unused imports .
Issue : The filesystem path of AF_UNIX sockets now uses the filesystem
: fix typo in turtle . rst .
Issue : Fix typo in NEWS entry
Move comments in configure . ac .
Cast bytes . Tanks Anthony .
ignore a test for package branch .
bpo - 32251 : sqlite3 docs ( GH - 7364 )
Merged revisions 72882 via svnmerge from
Merge
Simple error handling .
Merged revisions 82769 via svnmerge from
News for the tp_flags module .
- Got rid of WITH_CYCLE_GC
Merge with 3 . 3
Issue : Merge with 3 . 5
Merged revisions 82169 via svnmerge from
Issue : Some expensive the socket module to read on OpenSolaris .
Issue : Add zlib files necessary to run configure
Add two items
Added ALL analogous to rag_nextrange .
Merged revisions 62792 , 77266 , 77279 via svnmerge from
Issue : Fix test_regrtest for installed Python
On second though , NEXITFUNCS should be defined here and not in
Merged revisions 73870 , 73879 , 73899 - 73900 , 73905 - 73906 via svnmerge from
asyncio : sync with Tulip
Fix missing typos
Merged revisions 65152 via svnmerge from
Merged revisions 86281 , 86285 via svnmerge from
Use the DISTUTILS_DEBUG documentation .
Add a test for the HeaderParser class .
Patch : don', "'", 't define shebang lines
Patch : Add status code 421 to httplib .
Various minor markup nits .
Issue : MSVC ffi_prep_args doesn', "'", 't handle 64 - bit arguments properly
merge with 3 . 2
Merge Py Idle', "'", 's Rev 1 . 16 ( string methods )
Merged revisions 77798 via svnmerge from
bpo - 37726 : Correct threading UUID docs ( )
Minor markup .
Mark - fno - strict - aliasing when available .
Issue : Use the new _PyUnicodeWriter internal API to speed up str % args and str . format ( ) .
Merged revisions 76255 via svnmerge from
Lots of patch : don', "'", 't depend on the exact location of PythonCGISlave within the Python tree . ( jvr )
Bug : don', "'", 't test on Cygwin , as chmod doesn', "'", 't work reliably there
Fix suspicious markup
: add a note to audioop docs about signedness of sample formats .
Minor documentation changes relating to NullHandler , the module used for handlers and references to ConfigParser .
Closes : Merged fix from 3 . 4 .
Add a option for the HeaderParser module
Added note about __builtin__ . _ to section dicussing classes of
For Henry Gebel : Added docstrings for the', "'", 'bdist_rpm', "'", 'command , specifically
Merged revisions 67082 via svnmerge from
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
Fix Issue10793 - hashlib documentation issue on return type of digest
Issues , : Fix typos in the documentation and code comments
Tweak the goto length ( issue6723 )
Fix docstring in LocaleTime .
: merge with 3 . 2 .
: merge with 3 . 3 .
- Got rid of WITH_CYCLE_GC
Logical markup & other small nits .
Merge 3 . 5
A number of small adjustments .
Updated 3 . 8 . 0a4
test_directory ( ) : Remove the leftover temp directory that', "'", 's making
Fix versionchanged
Fix to improve the changelog filtering .
context managerify
Issue : Expose the POLLRDHUP constant in gzip . __missing__ .
bpo - 31709 : Fix gather to propagate cancel of itself with return_exceptions ( GH - 7209 )
merge 3 . 3
Added \\ sentinel { } for logical addressing .
Issue : add link to Jan Kaliszewski', "'", 's alternate constructor and ABC for named tuples .
NEWS for correct entry .
testing control chars and non - dict root objects
Delete pprint documentation .
Integrate the multifile . c
Merged revisions 80404 via svnmerge from
Use FSpCreateResFile ( ) in stead of CountMItems ( ) .
beef up docstring
: merge with 3 . 2 .
Merged revisions 78976 via svnmerge from
Merged revisions 76774 via svnmerge from
An interface to the font manager
Merged revisions 82492 via svnmerge from
Add missing setport comments .
Merged revisions 79866 - 79867 via svnmerge from
_dispatch ( ) : Don', "'", 't do a type comparison to see if it', "'", 's an
merge heads
Clarify the URLs
Merge with 3 . 4
Limit string size on one - character - strings . Fixes .
Issue : Fix a crash in the TextIOWrapper . tell method on 64 - bit platforms .
add security
- gusiprefs . c .
final round of __all__ lists ( I hope ) - skipped urllib2 because Moshe may be
take linkage def outside of WITH_THREAD conditional ( closes )
Merged revisions 60234 - 68119 , 68121 , 68123 - 68127 via svnmerge from
Issue : Deprecated silent truncations in socket . htons and socket . ntohs .
Merge Py Idle', "'", 's Rev 1 . 16 ( string methods )
[ Patch ] Rename has_key ( ) to handle regression ; private
Merge
Issue : Fix UnboundLocalError in AbstractDigestAuthHandler . get_algorithm_impls
: merge with 3 . 3 .
Fix : it', "'", 'fancy_getopt ( )', "'", 'so it resets the', "'", 'have_run', "'", 'flag for the
- Issue : Allow the _dbm extension to be built with db 4 . 8 . x .
Head merge
merge with 3 . 2
Issue : Fixed importlib . machinery . PathFinder to support
Check for existence of mremap ( )
Merge 3 . 4 ( test_urllib2net )
bpo - 33209 : End test ( )
whoops , I wanted that commented out by default , will add doc to Misc
Issue : BufferedReader . new_module ( ) now uses a subTest ( ) element which can
merge with 3 . 2
Input a4wide here
bpo - 22807 : Expose platform UUID generation safety information . ( )
post release updates
Remove sys_ * defines that are internal only and about to disappear
Use Tcl_SetObjResult instead of Tcl_SetResult .
Merged revisions 85400 via svnmerge from
: fix man page error in Misc / NEWS .
Issue : Add additional items
Issue : Add log2', "'", 's New ( asyncio . Task ) attribute to Pool .
Add news entry for .
Merged revisions 72079 via svnmerge from
merge heads
Combine the two seealso sections .
Add gc_collects to list of safe sys .
make plistlib module
Issue : Fix test_regrtest for installed Python
Add tests for __nonzero__ ( ) problems .
Issue : Remove support of RuntimeError
: fix typo in turtle . py .
Issue : Clarify the documentation of the 3 module .
Merged revisions 74675 via svnmerge from
Remove outdated entry from Misc / NEWS .
Issue : Make sure that the coefficient of a Decimal
: fix Windows test .
Issue : Fix a crash in io . BufferedIOBase . readinto ( ) when raw
Merge with 3 . 4
Send the tabnanny ( thanks to MH for noticing the problem )
Issue : Tweak last commit for this issue to pass in mode instead
Change the logic title to reflect the Tk affinity .
Add missing comma to clarify list of str docs .
rephrase
Fix two trivial bugs in the comment .
Add myself to fix of issue .
Merge with 3 . 4
posix_fildes ( ) : Remove the leftover temp directory that', "'", 's making
Merged revisions 67694 via svnmerge from
Document the proper exception to be raised by I / O operations on closed
* pythonmain . c : - k option , usage message , more environment flags .
bpo - 32107 - Improve MAC address calculation to fix test_uuid . py ( )
Merge .
Merge issue : don', "'", 't use sys . path in test_kqueue .
Merge with 3 . 5
Issue : test_faulthandler now uses sys . __stderr__
Fix issue : An errornous __length_hint__ can make list ( ) raise a
bpo - 30808 : Fix includes in main . c on Windows with non - MSC compilers ( GH - 13421 )
Fix typo .
Add link to the seealso section .
eval_code2 ( ) : In the extended calling syntax opcodes , you must check
Added WNOHANG
Document modeling manager for Coverity Scan .
Issue : Remove redundant doc field from PyType_Spec .
Always call the termination message for augmented assignment .
Issue : Fix deprecation warning in Lib / test / test_with . py
Introduced underlining of menu labels
Issue : Idle shell crash on printing non - BMP unicode character .
Be a bit more reversed .
bpo - 32362 : Fix references to non - existent multiprocessing . Connection ( ) ( GH - 6223 )
Change the generation of menu lists to show
Add key = argument to heapq . nsmallest ( ) and heapq . nlargest ( ) .
_Py_wreadlink ( ) uses _Py_char2wchar ( ) to decode the result , to support
oc / VS8 . 1 . 3 news .
cgi . FieldStorage . read_multi ignores Content - Length
Merged revisions 72905 via svnmerge from
Merged revisions 66383 via svnmerge from
Merge 3 . 5
merge with 3 . 3
Document the existence of expr ( ) and
Merged revisions 82301 via svnmerge from
Speed up and simplification
Added the - l info .
Merge 3 . 5 ( issue )
Issue : Replace FileIO', "'", 's quadratic - time buffer growth algorithm with a linear - time one .
formatfloat ( ) , PyErr_WriteUnraisable ( ) : Coverity found a cut - and - paste
Clean up distutils . dir_util
Fixes expected error when getting encoding while shutting down .
Issue : Fix importlib . test . import_ . test_packages to clean up
Issue : ast . dlopen ( ) accepts bytes , bytearray and str with
Add more TOC to the whatsnew index page .
merge 3 . 2
Merged revisions 71758 via svnmerge from
Issue : Use test_both ( ) consistently in test_importlib .
warn about the statvfs ( ) function .
base64 support by Sjoerd . Patch by David Rosenberg .
Patch : Fix regex problems with Cygwin .
Change the update_slot ( ) call outside the main loop to speed - up and simplify the block re - use logic .
test for correct bug , not property , but SUNWspro only but return
Issue : Merge from 3 . 5
Issue : Fixed crash when marshalling NotImplemented in . c .
bpo - 40275 : Fix pydoc ProactorLoopCtrlC ( ) ( )
Issue : Fix equality testing of dead weakref objects .
Separate isinstance
Merged revisions 59628 - 59422 via svnmerge from
Patch by Jp Calderone :
Merged revisions 80404 via svnmerge from
bpo - 34754 : Fix asyncio PidfdChildWatcher on waitpid ( ) error ( GH - 17477 )
Recompiled to fix version
Fixed defaultTestCase - > defaultTestResult
Try to fix test_gdb on s390x buildbots
Patch : FileInput now can be given an opening hook which can
email . xgtitle ( ) description : Add a missing " ) " from the response value
PyUnicode_EncodeUTF8 ( ) : tightened the memory asserts a bit , and at least
Issue : Fix validation of relative imports in parser module .
asyncio : pep8 - ify the code .
added concrete pystone " benchmark
Fix incompatible assignment for Coverity .
bpo - 5028 : Fix includes in main . c ( GH - 14218 )
bpo - 37148 : Fix test_time . test_thread_time ( ) ( GH - 10724 )
Issue : Merge with 3 . 5
bpo - 29822 : Unify implicit and explicit state in the frame and generator objects into a single value . ( GH - 20803 )
Make decimal . ContextManager a private implementation detail of decimal . localcontext ( )
bpo - 30329 : Catch Windows error 10022 on shutdown ( ) ( )
Bytes should never equal unicode .
Issue :', ['Issue : Use PyOS_string_to_double instead of PyOS_ascii_strtod in
Issue : os . utime ( ) now correctly resolves multiple nested symlinks on POSIX platforms .
Keep below 80 chars .
Issue : Allow 3 . 6 . 2 .
bpo - 41756 : Add PyIter_Send function ( )
Added notes about pimp and bundlebuilder to the Mac section .
Added a self - known file .
Issue : Fix a number of bugs in Tools / scripts / pindent . py .
* Bring lookkey ( ) and lookkey_string ( ) closer to dict version .
Issue : make it clear that create_default_context ( ) also enables hostname checking
bpo - 32468 : Retire pgen and use pgen2 to generate the parser ( GH - 11814 )
Issue : Update external link .
Fix markup .
Note that abspath ( ) was added in 1 . 5 . 2 .
Fix a problem of KeyboardInterrupt and see the test suite
Bug : document the probability density word to _mswindows .
Issue : Skip test_case_operation_overflow on computers with low memory .
Issue : Clarify ET . fileConfig ( ) and math . ntohs ( ) .
Issue : logging : fix bug in Unicode encoding fallback .
On SGI , we need to define _SGI_MP_SOURCE before including errno . h when
post release updates
Patch 1471925 - Weak linking support for OSX
Typo : " descrition " - - > " description "
: merge with 3 . 3 .
Better tests for bz2 to issue
Issue : Update htest check to unittest . py .
Merge 3 . 5
Issue : Merge from 3 . 6
: typo .
Added descrobject . c and mysnprintf . c
Merged revisions 71758 via svnmerge from
Issue : PYTHONWARNING no longer affects the run_tests . py script .
Don', "'", 't complain that non - existant registry entries cannot be deleted .
fix signature
Merged revisions 76156 via svnmerge from
Fix issue 26287 : While handling FORMAT_VALUE opcode , the top of stack was being corrupted if an error occurred in PyObject_Format ( ) .
Remove outdated note about constraining of the bit shift right operand . ( )
Merged revisions 88722 via svnmerge from
Removed THINK_C inclusion from inclusion .
Clarify description of the calendar module .
Harry Henry Gebel : When', "'", 'fancy_getopt ( )', "'", 'function into FancyGetopt
merge 3 . 3
Issue : Improve Filter documentation .
Issue : Stop the Latin - 32 decoder . It is now 3x to 4x faster
build _csv extension module
bpo - 36786 : Implement platform . AsyncExitStack . ( )
SRE 2 . 312 .
bpo - 31454 : Remove inaccessible videos from faq / 3 . 6 ( GH - 494 )
update pydoc test for 3 . 6 . 0rc1 .
Merge 3 . 2
Merged revisions 79866 - 79867 via svnmerge from
Fix a loooong - standing brainfart : I got the site - packages path for
Update the test for string objects .
Add a note reminding the reader that x are not sequences .
Issue : On Windows , increase the stack size from 2 MB to 4 . 2 MB to fix
Use Tcl_SetObjResult instead of Tcl_SetResult .
Removed THINK_C support .
typo ( patch : )
Issue : Fix ProactorEventLoop . close ( )
On SGI , we need to define _SGI_MP_SOURCE before including errno . h when
py - cvs - rel2_1 ( Rev 1 . 4 ) merge - move " import * " to module level
Merged revisions 86940 via svnmerge from
Issue : Raise OSError when marshalling too large object ( a sequence
Issue : Remove dead code
asyncio : sync with Tulip
Issue : Clarify kwargs parameter in sched . Event named tuple .
Merged revisions 68661 via svnmerge from
Issue : Clarify ET', "'", 's parsing of comments and processing instructions .
* pythonmain . c : - k option , usage message , more environment flags .
Minor fix for test_multiprocessing
Issue : Fix typo in NEWS entry
: add bullet versionadded to old docs .
Issue : Random generator objects now consume 2x less memory .
Use input ( ) directly that it is available .
merge 3 . 3
Merged revisions 61034 - 61036 , 68340 , 68368 , 61053 , 61055 - 61056 , 61061 - 61062 , 61066 , 61068 , 61070 , 61081 - 61095 via svnmerge from
Add version notes for name2codepoint ( ) , to be used by embedding .
Backport of patch : add docs for __objclass__ . py
Merge heads
Added missing \\ var { } for logical addressing .
Issue : Merge from 3 . 5
removed obsolete comment .
Issue : Fix Shift - JIS incremental encoder for error handlers different
one CERT_REQUIRED is enough
- Issue : Fix _PYTHON_PROJECT_BASE on Windows
: md5 weaknesses do not affect hmac , so remove the note about that .
Merge with 3 . 2
Issue : Fix rare test_signal / audio code execution in tests .
asyncore . xgtitle ( ) description contributed by Skip Montanaro .
Make test_base64 pass .
Issue : Merge rlcompleter change from 3 . 4 into 3 . 5
Input a4wide here
Issue : Fix IDLE for file early unable to encode
merge 3 . 5 ( )
The getstatus ( ) method of a Readcd instance always calls
Merge heads
Issue : Fix a crash in the TextIOWrapper . tell method on 64 - bit platforms .
Issue : Combine the doc pages for uploading and registering to PyPI .
Issue : multiprocessing : fix a race condition when a Pool is closed
Issue : Fix importlib . machinery . PathFinder to support
PyBuffer_New ( ) : Obsolete ; removed .
Issue : An modulefinder module now has less Clinic .
do_cmd_rfc ( ) : Simplify the generated markup .
Issue : Add IDLE configdialog tests to PC / python3 . def .
whatsnew : filecmp . DEFAULT_IGNORES ( )
Issue : Skip test_case_operation_overflow in test_asynchat .
whoops , I wanted that commented out by default , will add doc to Misc
Fix test_fnmatch .
bpo - 36341 : Fix test_master_read ( ) so that it succeeds on all platforms that either raise OSError or return b " " upon reading from master ( GH - 23536 )
Merge
Typo : " descrition " - - > " description "
: fix typo in the / throw / pop ( ) method .
Merged revisions 66162 via svnmerge from
rename args variable in CALL_FUNCTION to callargs ( avoids name
Merged revisions 70167 via svnmerge from
Added err_badcall ( ) docs .
condense import
Issue : Fix test_pdb on Windows
: document that Popen . returncode is set by Popen . poll / wait .
Update the comment and make it more robust ( I hope it had the problem )
Merge debug output removal with 3 . 3 .
Load the overview size .
Merged revisions 79866 - 79867 via svnmerge from
Issue : Improve the documentation and indexing of dict . __missing__ .
Don', "'", 't put < ioctl . h > unless on __VMS .
bpo - 40755 : Add rich comparisons to Counter ( GH - 20548 )
copy ( ) : Use augmented lists for MSL
Issue : Fix zlib . close ( ) on Windows for files larger than 2 GB
Issue : Fixed test_gdb failures on OS X .
Last Drake package .
Fix sporadic failure of test_pep277 on Windows : use support . rmtree ( ) instead of
SF patch : Replace push / pop clusters in ceval . c
Add empty section for bool
Issue : Fix Python tutorial
bpo - 36454 : Fix test_time . test_thread_time ( ) ( GH - 10724 )
Logical markup .
Issue : logging : Improved IronPython 2 . 6 compatibility .
Merge : : fix the Remove code nits .
Issue : Fix a couple of places where undefined behaviour can
Issue : Add a new PyFrame_FastToLocalsWithError ( ) function to handle
Catch ProtocolError can pass and bsddb3
Remove obsolete import .
Add change notes where im_class is discussed , since the exact meaning changes
merge 3 . 2 heads
Issue : Fix test_regrtest for installed Python
Merged revisions 77794 via svnmerge from
merge
asyncio : sync changes for Tulip
Issue : Merge from 3 . 5
Take out the " ARTICLE " to codec_list . c
Issue : Fix typo in Lib / test / test_with . py
Issue : Improve error message when pass a non - iterable as
Merged revisions 79316 via svnmerge from
bugfix for * BSD platforms . ( oops messed up # define )
merge 3 . 5
Merge with 3 . 4
bpo - 38870 : Expose AsyncMock and condense blurbs ( GH - 7500 )
Don', "'", 't store unexpected errors in PyErr_WarnExplicitObject ( ) . ( )
Merge 3 . 5 ( odict )
DECREF pi module
New tool .
Issue : Added tests for CLI of the zipfile module .
Added newind . py and mysnprintf . fsum
Fix typo
Issue : Update OS X installer welcome and readme files for 3 . 4 . 2 .
Customary 2nd submit of this file after it', "'", 's been changed .
asyncore . xgtitle ( ) description contributed by Skip Montanaro .
Merge from 3 . 5
Misc / NEWS entries for zipapp changes
remove unused variables .
Fix Armin', "'", 's bug 1333982 . He found it , he didn', "'", 't created it : - )
Merge 3 . 4 ( asyncio )
Updated for issue
Issue : Merge from 3 . 5
Add bigmemtest decorator to test of issue .
Issue : Fix uint_converter ( )
Add back Raymond', "'", 's interact ( ) function .
Issue : replace assertions in test_os .
: merge with 3 . 3 .
: merge with 3 . 3 .
1 . 0 . 3 final .
Merge doc fixup from 3 . 5
Merge Py Idle changes :
Make the stdlib test suite clean up .
Merged revisions 78576 via svnmerge from
Add some VM developers .
Issue : The modulefinder module now supports extended opcode arguments .
Fix two formatting
asyncio : Add support for log1p ( ) , PyMem_Calloc ( ) ,
merge heads
Document the user to expr ( ) function .
Added a new test for :
Improve / remove mentions of functions gone in 3 . x
- Got rid of PopUp', "'", 's patch
Fix NULL check in sock_sendmsg_iovec . CID 1372885
Issue : Add an debug PyFrame_FastToLocalsWithError ( ) function to handle
From the changes of the poplib module
Merge heads
Added missing static option for OptionMenu . Issue .
Merged revisions 75634 via svnmerge from
Try to fix word breakage in r78384 .
Issue : Fix OSError decompressor handling of unused_data with multiple calls to decompress ( ) after EOF .
Issue : Added tests for the spwd module . Original patch by Vajrasky Kok .
Fix problem introduced by Barry : if you hit ^ C to
Closes : clarify documentation for random . literal_eval ( ) .
Issue : Removed redundant checks leftover from int to long conversion
Merged revisions 78722 via svnmerge from
Fixed : optimise code for all dbm . stat doesn', "'", 't exist on MacOS9 .
tag 2 . 7 alpha 1
1 . Fail Sourceforge checkin , idle - dev posting
- general markup cleanup
_Py_wreadlink ( ) uses _Py_char2wchar ( ) to decode the result , to support
Typo
: make linecache . crc32 ( ) work with the iterable file
Patch : AF_NETLINK sockets basic support .
Issue : The binascii module now uses Argument Clinic .
Fix error handling .
bpo - 1635741 : Port _overlapped module to multi - phase init ( PEP 489 ) ( GH - 22049 )
Merge from 3 . 3
Issue : Remove outdated versionchanged directives .
Issue : IDLE now uses sys . __stderr__
Merged revisions 72260 via svnmerge from
Issue : Fix raiseExceptions typo in the tests .
Enable _AH module .
small FAQ for Python CVS and patches at SourceForge
Make decimal . ContextManager a private implementation detail of decimal . localcontext ( )
Issue : distutils . extractfile ( ) failed to resolve symbolic links when
Mostly minor changes .
bpo - 30642 : IDLE : Fix test_query refleak ( )
Issue : Fix', ['( py - indent - seeding ) : Simplification .
Merged revisions 83966 via svnmerge from
Issue : Fix a comment in the TextIOWrapper . dumb
bpo - 15221 : Update os . path . is * ( ) documentation ( GH - 5185 )
win_getpass ( ) : Merge in __members__ and __methods__ too ( if they exist ,
Merge with 3 . 4
Issue : Keep the POLLRDHUP decoder in audioop module .
Whitespace normalization .
* Python / pythonrun . c
Issue : fix link
Added reverse lookup
Whitespace normalization .
Move the overview to the low - itself compile .
Clean up some markup .
Merged revisions 72306 via svnmerge from
merge 3 . 5
Add support for 85cb90f79cbf , precision in the ssl module .
Merged revisions 75519 via svnmerge from
Fix typo in markup
merge with 3 . 4
Added an open_pathname ( ) method which opens a resource file by pathname ,
Stylistic changes to the bz2 module to make it easier to transplant changes from
Typo fix
merge heads
Issue : Expose unicode_writer_prepare ( )
Added example
Fix typo in comment .
Merge 3 . 5 ( test_urllib2net )
Fix typo in comment
Use __builtin__ instead of USE_UCS4_STORAGE and Py_UNICODE_SIZE
Update bug tracker for OS X .
Merge 3 . 4 ( test_urllib2net )
Issue : Random read objects now consume 2x less memory on 64 - bit .
Merged revisions 68302 via svnmerge from
merge
Issue : Fix faulthandler_thread ( ) : release cancel lock before join lock
Add a test for the HeaderParser class .
fix code .
- general markup cleanup
Refer to __builtin__ . file , not __builtins__ . file
: fix test on Windows .
Issue : The fromhex ( ) class methods of bytes and bytearray subclasses
Issue : Fix test_signal on Windows
Issue : TarFile . extractfile ( ) failed to resolve symbolic links when
Added missing decref
bpo - 8538 : Add support for boolean actions to argparse ( GH - 11478 )
Issue : Make sure that the coefficient of a Decimal
: fix issue introduced .
Fix markup
merge heads
: fix doctests in Doc / howto / regex . rst . Patch by Chris Jerdonek .
Add entry for curses . panel documentation .
Fix incompatible assignment warning .
bpo - 33499 : Add what', "'", 's new entry for os . errno removal ( GH - )
Issue : Fix bytes and AST docstrings
bpo - 9216 : Nobody expects the geohashing FIPS inquisition ( GH - 19520 )
Fixed defaultTestCase - > defaultTestResult
bpo - 22807 : IDLE : Fix test_query refleak ( )
bpo - 40941 : Unify implicit and explicit state in the frame and generator objects into a single value . ( GH - 20803 )
Mods by Just :
Fix site - mail defaultTestResult
Merged revisions 73880 via svnmerge from
PyClass_New ( ) : Apply the extended Don Beaudry hook back in . When one of
Added \\ label { } to logical addressing .
: typo .
Merge 3 . 5
Samuel L . Bayer :
Fix markup .
Merged revisions 81275 via svnmerge from
merge 3 . 5 ( )
Issue : Fix Shift - JIS incremental encoder for error handlers different
Fixed issue : CVE - 2012 - 0876 : Randomize hashes of xml attributes
Fix SF bug # 561858 Assertion and patch written by diana .
Take PyUnicode_TransformDecimalToASCII out of the limited API .
Update the URLs
Add a note that apply ( ) is needed since the extended call syntax is
On second though , NEXITFUNCS should be defined here and not in
test_os : report the specific function for teardown ( )
Use correct format specifier for Py_ssize_t variable to PyArg_ParseTuple ( ) .
merge with 3 . 3
Fix test_urllib2 . py .
Issue : Add SHA - 3 . SHAKE support to hashlib module .
whoops , I wanted that commented out by default , will add doc to Misc
Issue : Fix segfault in AbstractDigestAuthHandler . get_algorithm_impls
Merged revisions 79866 - 79867 via svnmerge from
Try to fix test_gdb on s390x buildbots
Fix bug introduced in r68451 : stdio must always be opened in line - buffered mode
bpo - 39943 : Fix pymain_init_pycache_prefix ( ) ( )
Add news entry for 57f0af61da53 .
Issue : Cleanup unicode_writer_prepare ( )
Removed reliance on gcc / C99 extension .
Issue : Optimize the socket module to Argument Clinic .
Merged revisions 78322 via svnmerge from
The usual
bpo - 34913 : Add _PyPreConfig . _config_init ( GH - 13481 )
Document set_unittest_reportflags ( ) .
bpo - 28055 : Fix unaligned accesses in siphash24 ( ) . ( GH - 6123 )
Add test for issue .
Beef - up docs and tests for itertools . Fix - up end - case for product ( ) .
merge heads
SRE Nicolas
Add missing setport size to mkdistr_ui , and added scrap for new compatibility
Merged revisions 70167 via svnmerge from
Issue : CVE - 2013 - 1752 : Change use of readline in imaplib module to limit
Regenerated heads
Added a note that apply ( ) is needed since the extended call syntax is
Color rid of new support for itertools .
What', "'", 's New in 3 . 3 : reorder modules
Fix test_datetime on Windows ( 2006 - 08 - 10 fixed the actual locking issue ) .
Merge 3 . 3
Issue : Use surrogate - related macros
Add Amaury Forgeot d', "'", 'Arc .
Fix docstring of types . readlines ( ) .
Issue : Fix shell for IDLE build in Idle version .
Merge : : fix pydoc . encoders . encode_7or8bit when applied to binary data .
bpo - 37731 : Optimize _Py_hashtable_get_ptr ( ) ( GH - 20066 )
Merged revisions 78297 , 78308 via svnmerge from
Issues , : Fix typos in the documentation and code comments
merge with 3 . 2
Add examples to list of functions from chapter .
Added docs for shlex . vcproj
Merged revisions 73931 via svnmerge from
PyTuple_New ( ) : vrbl i no longer referenced , so removed it ( which kills
Whoops , it', "'", 's late - - generated output now matches what', "'", 's checked in . . .
Trent Mick < trentm @ activestate . com > :
Add a few more BringToFont calls
Bug : add support for log1p ( ) to describe handle
Don', "'", 't hide unexpected errors in PyErr_WarnExplicitObject ( ) . ( )
Take out constant
Socket handler closed prior to end of test .
Various edits
Issue : Merge from 3 . 6
Issue : test . support . run_doctest ( ) doesn', "'", 't change sys . stdout .
Fix warning in a few minor nits .
bpo - 33652 : Distinguish 32 and 64 - bit user site packages on Windows ( GH - 22098 )
Merged revisions 77798 via svnmerge from
Added', "'", 'get_platform ( )', "'", 'to construct a string that identifies the current
use isinstance
typo
Issue : Prevents zipimport from unnecessarily encoding a filename ( patch by Eryk Sun )
Issue : Merge 3 . 4 into 3 . 5
merge
Issue : Cleanup unicode_writer_prepare ( )
Issue : The asynchat . async_chat . handle_read ( ) method now ignores
Issue : Merge 3 . 6
Merged revisions 72079 via svnmerge from
bpo - 29822 : _PyGILState_Init ( ) returns PyStatus ( GH - 18908 )
( Merge 3 . 4 ) Issue : Fix an unicode error in the pydoc pager when the
Merge with 3 . 3
Issue : Fix faulthandler_thread ( ) : release cancel lock before join lock
: Remove misleading code from weakref_segfault ( ) .
Merge from 3 . 5
eval_code2 ( ) : In the extended calling syntax opcodes , you must check
Somehow , I wanted this commented out , so that', "'", 's unreliable because set
Derivative of SF patch 423181
merge heads
Add loaded lookup
bpo - 15221 : Update os . path . is * ( ) documentation ( GH - 5185 )
Merge Py Idle', "'", 's Rev 1 . 16 ( string methods )
asyncio : Fix CoroWrapper for asyncio . Lock ( ) .
- general markup cleanup
Initial revision
Issue : Oops , the patch for codecs . encode / decode doc was written by
Merged revisions 72079 via svnmerge from
Issue : Raised OverflowError if % c argument is out of range .
Merge with 3 . 2
Issue : os . utime ( ) now correctly resolves multiple nested symlinks on POSIX platforms .
bpo - 22490 : Remove __PYVENV_LAUNCHER__ from environment during launch ( GH - 9516 )
Merge heads
bpo - 33666 : Improve the text of re . __contains__ ( GH - 2525 )
Merged revisions 84810 via svnmerge from
Issue : Cleanup unicode_writer_prepare ( )
Add links from builtins module docs to built - in functions and constants docs
Python 3 . 8 . 0a4
Issue : Add PyType_GetSlot .
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
bpo - 30600 : Fix test_flush_return_value on FreeBSD ( GH - 9451 )
bpo - 38449 : Fix crash in call_readline ( ) ( GH - 728 )
Issue : Update Windows documentation from 3 . 6 .
bpo - 28682 : Add Py_DEPRECATED to deprecated unicode APIs ( GH - 20878 )
Variant of SF patch 423181
: first permissions for legacy filename .
Merged revisions 82169 via svnmerge from
Issue : On Windows , use mbcs codec in strict mode to encode and decode
Add HAVE_MKTIME
: fix typo in init . rst .
avoid casting with this nice macro
bpo - 38449 : Improve the code for each __classcell__ . ( GH - 12715 )
Jack Jansen : Removed support for long - dead Think C compiler
Simple size of threaded embedding
whatsnew / 3 . 5 : Add missing word " class "
Issue : Fix distutils . sysconfig . import_ . test_packages to clean up
Fix PyUnicode_CopyCharacters ( ) doc
Issue : Fix a test_signal failure introduced by the buildbots in
: merge with 3 . 3 .
Issue : Fix test_regrtest for installed Python
PyTuple_New ( ) : vrbl i no longer referenced , so removed it ( which kills
( py - compute - indentation ) : Simplification ) typo
Various edits
Added ResObj_OptNew and - python2 . h to macglue . h in
Merged revisions 82301 via svnmerge from
Merge 3 . 3
Minor changes . Explain that for class exceptions , use excdesc but do not
Merged revisions 82492 via svnmerge from
when print ( ) gets unicode arguments , sep and end should be unicode by default
Merged revisions 83442 via svnmerge from
Issue : Fix test_signal on Windows
Issue : Speed up unittest . mock ( ) documentation .
Document the existence of PyUnicode_InternInPlace ( ) and PyOS_setsig ( ) .
Merged revisions 68221 via svnmerge from
Fix suspicious markup
GetoptError is always initialized with exactly two parameters , so simplify
: fix broken links in Misc / NEWS .
Merged revisions 80179 via svnmerge from
Issue : Fix uint_converter ( ) in zlibmodule . c , fix the " > UINT_MAX " check
Issue : Fix distutils tests on Windows ( OS X ) .
Merged revisions 77798 via svnmerge from
Merged revisions 68221 via svnmerge from
Fix typo .
Disable pythons version
Reflow dataclasses and popen2 . py changes .
eval_code2 ( ) : In the extended calling syntax opcodes , you must check
autoreconf : configure - > defaultTestResult
merge 3 . 2
Issue : Merge from 3 . 5
bpo - 33319 : Update references to RFC3339 and ISO8601 from the logging documentation . ( GH - 7297 )
Merged revisions 80990 via svnmerge from
bpo - 22490 : Remove __PYVENV_LAUNCHER__ from environment during launch ( GH - 9516 )
Merged revisions 68276 via svnmerge from
: remove Py_GetBuildNumber ( ) , which was removed in favor of Py_GetBuildInfo ( ) .
Fix typo .
explicitly cast away constness to silence compiler warning
Various nits
use isinstance
Fix markup .
Fix by reverting revision 80761 .
bpo - 31825 : Fix OverflowError in call_readline ( ) ( GH - 728 )
Teach the peephole optimizer to fold simple constant expressions .
Final touch - ups on users .
bpo - 36502 : Correct documentation of EventLoop . isspace ( ) ( GH - 15019 )
Issue : Fix call_function ( ) : release cancel lock before join lock
Remove outdated file
Add a test for .
Issue : Fix typos in the ssl module .
Remove unused var
Added zlib
Oops , fix a typo in the comment
Issue : Some binascii . FileIO . __all__ anymore
: merge with 3 . 2 .
Merged revisions 86985 via svnmerge from
Wire , the makefile ( ) function : add a missing " ) " from the response value
Issue : Fix PyDict_New ( ) to handle correctly new_keys_object ( ) failure
Branch merge
Minor tweek doesn', "'", 't exist on MacOS9 .
Issue : Add missing import to __all__ for Decimal module .
Fix weird whitespace
Use CountMenuItems ( ) in stead of CountMItems ( ) .
Fix the comment in the previous checkin .
typo
More new project for the separate ssl .
Add a usegmt flag to the enable API
Add missing comma .
Upgrade the news entries .
target libinstall :
Enable the goto option for distutils options .
Add missing decref
Update to MvL', "'", 's patch to avoid returning 2 when tp_compare
take linkage def outside of WITH_THREAD conditional ( closes )
On second though , NEXITFUNCS should be defined here and not in
Merge 3 . 5 ( odict )
Issue : Use assertWarns in test_asynchat .
bpo - 26506 : subprocess can use posix_spawn with pipes ( GH - 11575 )
Rip out download files .
Use floor ( fatal function ) to be raised with OSError .
Issue : Raise ValueError when marshalling too large object ( a sequence
Issue : Fix collapsing of RuntimeError
merge heads .
Fix mismerges of version directives .
Add ACKS entry for 57f0af61da53 .
Issue : The modulefinder module now uses base filename instead of displaying
Issue : PyType_Ready ( ) now checks that tp_name is not NULL .
SF patch : Replace push / pop clusters in ceval . c
Add mimetypes documentation to library reference dependencies .
take linkage def outside of WITH_THREAD conditional .
PyTuple_New ( ) : vrbl i no longer referenced , so removed it ( which kills
Fix glaring bug in get_magic
fill_number ( ) ensures that the', "'", 'digits', "'", 'string is ready
Re - word
Fix issue 26287 : While handling FORMAT_VALUE opcode , the top of stack was being corrupted if an error occurred in PyObject_Format ( ) .
Issue : Add log2 function to math module . Patch written by Mark
Added S_IMODE ( ) to strip the file type ( S_IFMT ) from a mode .
use skipUnless
merge 3 . 5 ( )
bpo - 28298 : make inspect . isabstract ( ) work during __init_subclass__ . ( )
Add test to verify that the LazyList - based
Merged revisions 66383 via svnmerge from
merge with 3 . 2
Merged revisions 67694 via svnmerge from
Revert stdout . vcproj
Issue : Oops , subprocess : fix bug in AbstractDigestAuthHandler . get_algorithm_impls
Typo fix .
Exercise xrange files from mailer daemons .
Grammar fix .
Fix spacing .
Merged revisions 68276 via svnmerge from
( Merge 3 . 4 ) Issue : Fix __qualname__ ( ) docstring
Issue : Dict reduces possibility of 2nd conflict in hash table .
bpo - 36373 : Fix deprecation warnings ( GH - 15889 )
Issue : Document FileIO . dlopen ( ) to handle error if the file
Fix PyUnicode_CopyCharacters ( ) doc
merge 3 . 4 ( )
fix refleak in the error case
Issue : Dict reduces possibility of 2nd conflict in hash table .
Issue : Add configure . get_magic ( ) function to implement
Branch merge
Register IterableUserDict as uid instead of indenting them
Merge with 3 . 4
Issue : Fix test_regrtest in AbstractDigestAuthHandler . get_algorithm_impls
Removed math . h
err is no longer used
Added getfiletype routine
Fix typo .
bpo - 23427 : Add Py_DEPRECATED to deprecated unicode APIs ( GH - 20878 )
Issue : Document exception when marshalling is used .
Merge with 3 . 2 grammar fix
Merge with 3 . 4
Fix issue : When the urllib2 of a Decimal
Merged revisions 85400 via svnmerge from
code_repr ( ) , com_addop_varname ( ) , com_list_comprehension ( ) ,
Made DateTime', "'", 's constructor accept a time . struct_time class ,
Removed THINK_C support .
Merge issue : Fix bisect unittest . Patch by Matheus Vieira Portela .
Issue : PyType_Ready ( ) now checks that tp_name is not NULL .
Merged revisions 82169 via svnmerge from
Issue : Extract ffi_prep_args clinic is None with ioctl .
bpo - 41627 : IDLE : Fix test_query problems on OS X ( )
merge with 3 . 3
handle_system_exit ( ) flushs files to warranty the output order
SF bug : Explain to make an exception that it
bpo - 26133 : Fix test_asyncio ProactorLoopCtrlC ( GH - 14074 )
Issue : Fix faulthandler_thread ( ) to handle correctly new_keys_object ( ) failure
Merged revisions 75143 via svnmerge from
bpo - 28055 : Use InvalidURL instead of ValueError . ( GH - 13044 )
Re - generated with slightly more intelligent script
Fix typo .
fix markup typo
bpo - 31843 : sqlite3 . connect ( ) now accepts PathLike objects as database name ( )
patch : don', "'", 't define _XOPEN_SOURCE on Mac OS Panther
- Regenerate PRINT_ITEM ( ) .
Merge from 3 . 5
Added PyMac_Initialize ( ) routine , which is like grep / egrep / emgrep but uses Perl
Merged revisions 83442 via svnmerge from
Issue : Drop unused check
Only install the arr variable instead of a terminal .
Issue : Merge 3 . 6 .
Merged revisions 86281 , 86285 via svnmerge from
Issue : Clarify that invoking freeze_support ( ) in argparse .
Much rewritten . Added Win32 registry stuff to PyNumber_Add ( ) .
Issue : multiprocessing . get_versions ( ) encodes unicode message to
Patch : close sockets properly in urllib2 .
Fix sporadic warning in xxlimited . py . ( GH - 21477 )
Head merge
Issue : CVE - 2013 - 1752 : Randomize hashes of xml attributes
Put a Py_IS_NAN call for script_from_examples ( ) .
Fix incorrect stacklevel for DeprecationWarnings originating from the struct module .
Issue : Update the POLLRDHUP API documentation .
Fix typo
Richard Wolff', "'", 's additional changes ; some layout nits , and add the
Added FOUR_CHAR_CODE define
Merged revisions 60234 - 60244 via svnmerge from
Added new - style option to open ( ) and PyList_SET_ITEM ( ) .
correct len signature in docstring ( closes )
bpo - 34019 : Fix wrong arguments for Opera Browser ( )
Added reverse lookup
Issue : Fix a test_signal failure with - Wstrict - overflow .
bpo - 36710 : Add _PyPathConfig_ComputeArgv0 ( ) ( )
Fix waiter cancellation in urllib2 .
Tighten OverflowError failure .
Patches describing the statvfs ( ) and fstatvfs ( ) functions . Additional
Try to fix test_gdb on s390x buildbots
Add test for 85cb90f79cbf and see how the code handles all flags at once
merge with 3 . 3
Revert issue 2014
Remove outdated comments from yes .
Merged revisions 65152 via svnmerge from
Clean up some confusing text left by PROTOCOL_SSLv23 - > PROTOCOL_TLS transition ( )
* wdbframewin . py ( re_eval ) : set __privileged__ in globals so private
- merge 3 . 5
bpo - 32037 : Use the INT opcode for 32 - bit integers in protocol 0 pickles . ( )
add a autobuild - quick target that only rebuilds HTML .
Use PyModule_AddIntConstant ( ) instead of creating a private helper .
For the escape ( ) function , added a reference to the quoteattrs ( ) function
Issue : Fix line of file execution unable to encode
bpo - 37731 : Reorder includes in xmltok . c ( GH - 12442 )
resolve issue 2014
Patches describing the statvfs ( ) and fstatvfs ( ) functions . Additional
Initialize of FakeSocket
: fix refleak in kqueue , occurring in certain error message
Fix refleak .
Remove uneeded comment .
handle_system_exit ( ) flushs files to warranty the output order
: fix typo in init . rst .
Updated the blocksize ( ) function consistently .
A number of small adjustments .
Update the __version__
bpo - 37571 : Remove Date and Release field in whatsnew / 3 . 6 ( GH - 494 )
Issue : Fixed zlib on Windows
merge heads
Merged revisions 73109 via svnmerge from
Use isinstance
Merged revisions 73871 via svnmerge from
Add some VM developers .
Make decimal . ContextManager a private implementation detail of decimal . localcontext ( )
Issue : Fix long overflow corruption on OS X with an SDK .
Merge 3 . 5 ( odict )
Changed comments in beta1 news
Fixed - up tests for multiple Tkinter , which this
Issue : Fixed a gcc warning with - Wstrict - overflow .
Add notes about doco
Issue : Fix logging . fullmatch ( ) on Windows for files larger than 2 GB
Issue : Merge subprocess . Popen ( ) documentation improvements from 3 . 2 .
Issue : Add docstrings to Jan Kaliszewski', "'", 's alternate constructor and ABC for named tuples .
Don', "'", 't hide unexpected errors in PyErr_WarnExplicitObject ( ) . ( )
Issue : Fix the crash in the parser module when handle
Fix typo in comment
PYMALLOC_ { CLEAN , DEAD , FORBIDDEN } BYTE symbols : remove the PYMALLOC_
Now that there are no longer used
Merge
eval_code2 ( ) : In the extended calling syntax opcodes , you must check
Bug : don', "'", 't test on Cygwin , as chmod doesn', "'", 't work reliably there
Issue : test_faulthandler now uses sys . __stderr__
Issue : Fix faulthandler_thread ( ) of ceval . c to handle PyTuple_New ( ) failure
Merged revisions 79161 via svnmerge from
GetoptError is missing
Issue : improve htest and sysconfig method .
: merge with 3 . 2 .
code_repr ( ) , com_addop_varname ( ) , com_list_comprehension ( ) ,
Merge 3 . 4 ( odict )
bpo - 339827 : Do not swallow exceptions in the _ssl module . ( GH - 12756 )
bpo - 40401 : Remove nuspec pyhash . h include from pythoncore . vcxproj ( GH - 19725 )
long_scan is no longer used
Merge heads .
Merge 3 . 4 ( test_urllib2net )
Merged revisions 81881 - 81882 via svnmerge from
: fix distutils . loads ( ) error message .
Issue : Clarify that descriptors get used in an * owner * class or one of its parents .
Issue : Fix test_warnings on Windows
email . compat32 - message . rst : Fix typo in the word ` message ` ( GH - 379 )
Add ACKS entry for 57f0af61da53 .
: merge with 3 . 3 .
new_arena ( ) : In the extended calling syntax opcodes , you must check
Merge 3 . 2
whoops , I wanted that commented out by default , will add doc to Misc
Fixed issue11082 - Reject str for POST data with a TypeError . Document the need to explicitly encode to bytes when using urlencode .
Fix test_fnmatch .
Documentation for the enumerate ( ) function .
bpo - 40010 : Optimize pending macros in multithreaded applications ( GH - 19091 )
Add PyExc_NotImplementedError
Close : Clarify documentation for Handler . handleError .
libpython . py : defer call to gdb . lookup_type (', "'", 'PyUnicodeObject', "'", ')
bpo - 33176 : Add asyncio . AsyncExitStack . ( GH - 12132 )
delta loading and simplification
Interface to Mac Communications Toolbox ( only Connection Manager for
Moved PyObject_ printout
bpo - 36373 : Fix pydoc ProactorLoopCtrlC ( ) ( )
merge from 3 . 2
Remove obsolete modules
Issue : Use assertWarns instead of PyOS_ascii_strtod in
Issue : Fix test_warnings for locale encoding unable to encode
Issue : Use local file . Patch by Phil Webster .
Let debug printout
Issue : Fix test_signal . py . import_ to clean up
merge
Merged revisions 81859 via svnmerge from
Merged revisions 76255 via svnmerge from
Add a note about pow ( x , y ) equalling x * * y ( the " * * " operator
Mark - fno - strict - aliasing if available .
Issue : Fix test_gdb failures on KeysView and ItemsView .
Merged revisions 80404 via svnmerge from
( Merge 3 . 4 ) Issue : Fix an unicode error in the pydoc pager when the
target libinstall :
: fix _dummy_thread import name
Update .
merge heads
Merged revisions 79866 - 79867 via svnmerge from
ignore control - l ( parallelling change to tokenizer . c )
: .
Don', "'", 't complain that non - existant registry entries cannot be deleted .
Merged revisions 72882 via svnmerge from
Issue : Correct cross - references in the tutorial .
Merged revisions 86940 via svnmerge from
Merged revisions 84289 via svnmerge from
Add the calendar synopsis .
Reject float as uid or gid .
Add comments explaining IDLE library library .
Issue : various authorizer constants in sqlite3 docs .
Issue : Add - - timeout option to regrtest ( disabled by default ) .
Stylistic tweaks to the bz2 module to make it easier to transplant changes from
Issue : Fix a leak of file descriptor in the dbm . dumb
Issue : Re - implement imp . cache_from_source ( ) in Lib / imp . py .
Merged revisions 79344 , 73872 via svnmerge from
Merged revisions 77877 via svnmerge from
- Fix the broken link for pylons project .
Merge 3 . 5
merge 3 . 5 ( )
Merge for issue
Clarified how to detect build warnings
Merged revisions 66539 via svnmerge from
Make check_warnings error messages more informative
Merged revisions 68276 via svnmerge from
Sigh . Strengthen the resriction of the previous checkin : tp_new is
merge 3 . 3
Merge with 3 . 4
bpo - 38982 : Fix asyncio PidfdChildWatcher on waitpid ( ) error ( GH - 17477 )
Duncan Grisby : Don', "'", 't do _XOPEN_SOURCE on Windows
Remove outdated example .
Anthony Baxter < anthony @ interlink . com . au > :
Merged revisions 82169 via svnmerge from
use \\ citetitle and \\ programopt .
Merged revisions 84366 via svnmerge from
Merged revisions 68221 via svnmerge from
Treat None as a constant .
Add tests for VxWorks ( happens in pybench ) .
Fixed the - - with - cxx - main build .
Optimization style from the new
Patch by kragen @ pobox . com : When tracing is turned on , lines shorter
bpo - 34519 : Add _PyPreConfig . _config_init ( GH - 13481 )
PEP 489 : fix dupe in functions and show / types docs
Reject float as uid or gid .
Added _PyBytesWriter_WriteBytes ( ) to factorize the code
Use US paper documentation .
: merge with 3 . 3 .
Issue : merge from 3 . 5
Fred Gansevles < gansevle @ cs . utwente . nl > :
Whitespace normalization .
Merged revisions 66539 via svnmerge from
Some rearrangements for bz2 . vcproj
Typo : " descrition " - - > " description "
Version upped . Will backport to 2 . 0 .
bpo - 33383 : Fix test_master_read ( ) ( GH - 7957 )
Issue : Update the POLLRDHUP ( ) and _Py_write ( ) FAQ .
Use correct format specifier for Py_ssize_t variable to PyArg_ParseTuple ( ) .
bump patchlevel to 2 . 0c1
Merged revisions 79866 - 79867 via svnmerge from
Use INSTALL_PROGRAM for makesetup
Note that the Unicode codec is only capitalized .
markup adjustments
Getting rid of macfs .
Fix a typo .
Optimize write ( ) and lookkey_string ( ) closer to tuple version .
: fix broken links in the docs .
removed two unneeded lines from PyObject_Compare ( ) .
Revert rewritten and diff .
bpo - 33804 : Add status string 421 to http . HTTPStatus ( GH - 2589 )
Issue : Fix ElementTree . fullmatch ( ) on Windows for files larger than 2 GB
Issue : Merge from 3 . 5
RFE : Socket objects', "'", 'family , type and proto properties are
Fix indentation
Issue : Clarify token . OP handling rationale in tokenize documentation .
Merge with 3 . 4
unicode_subtype_new ( ) uses _Py_char2wchar ( ) to decode the result , to support
Merge default
typed_subpart_iterator ( ) : When getting the main type use', "'", 'text', "'", 'as the
Merge from 3 . 3
Issue : Fix Python 2 . 5 compatibility in NEWS
Issue : The binascii module now uses Argument Clinic .
bpo - 41302 : Add PyInterpreterState . runtime field ( GH - 17270 )
Merge tag', "'", 'v3 . 10 . 0a3', "'
Issue : Fix a crash in the TextIOWrapper . tell method on 64 - bit platforms .
: fix refleak in kqueue , occurring in certain error conditions .
NEWS for existence of mremap ( )
Issue : Fix subprocess example on Windows in asyncio doc
Issue : Fix incorrect precedence of signals in Decimal module .
Add notes pointing out that these classes are kept for backward compatibility
bpo - 34373 : fix test_mktime and test_pthread_getcpuclickid tests on AIX ( GH - 8726 )
Remove obsolete comment ( yes , build_ext supports C + + )
bpo - 33338 : [ c ] Minor code cleanup ( )
Bug : Fixing SRE parser to handle', "'", '{ }', "'", 'as perl does , rather than
Impl ssize_t
Issue : Merge with 3 . 5
 add support for Steven operators for the', "'", 'bdist_rpm', "'", 'command , specifically
only clear for python - O
Issue : Fixed a failing corruption in ossaudiodev module .
Add todo scopes to the improved API
Issue : Fix a crash in str . decode ( ) when error handler returned
tag 2 . 7 alpha 1
Fix test_pty on OSF / Tk ( 2006 - 08 - 10 fixed the actual locking issue ) .
Initial revision
found_terminator ( ) : Fix indentation
Fix waiter cancellation in test_concurrent_futures ( closes )
Merged revisions 68381 via svnmerge from
Issue : Renamed WinFireFox to WinFirefox
Merge
remove more space .
Patch : Implementation of major , minor and makedev .
Added notes to list comprehensions
Instead of these labels
Issue : The modulefinder module now supports processing for binary mode
Merge
Fix posixpath . realpath ( ) for multiple pardirs ( fixes issue ) .
Adjust the content of the generation HTML .
Merged revisions 78576 via svnmerge from
Merged revisions 55184 - 84409 via svnmerge from
Added notes about the linuxaudiodev and ossaudiodev modules .
new_arena ( ) : In the extended calling syntax opcodes , you must check
merge
SF : merge from release24 - maint branch : add useful read - only
Teach the peephole optimizer to fold simple constant expressions .
Added zlib
Typo fix .
Merged revisions 75224 via svnmerge from
Fix markup .
The usual
Add a few items
Merged revisions 83442 via svnmerge from
Issue : Prevent equivalent of UNICODE macros in cPickle .
Merged revisions 72787 via svnmerge from
Merged revisions 80990 via svnmerge from
Merged revisions 77798 via svnmerge from
Add Samuele Pedroni
Issue : Re - implement imp . cache_from_source ( ) in Lib / imp . py .
Don', "'", 't declare PY_WITH_KECCAK
Added a - - ciphers option to Lib / test / ssl_servers . py
merge 3 . 3
Fix : Rework the error message when PyObject_RichCompareBool ( ) returns a NULL pointer .
Closes : merged fix from 3 . 2 .
merge
Issue : Merge from 3 . 5
Issue : Add a note to whatsnew / 3 . 5 . rst for nested functools . partial calls
bpo - 33209 : End framing at the end of C implementation of pickle . Pickler . dump ( ) . ( GH - 6363 )
Include site - packages directory .
Issue : IDLE : fix bug in Unicode encoding fallback .
Merge 3 . 4 ( test_urllib2net )
: fix typo in kqueue , occurring in certain error message
Issue : Fix test_regrtest for installed Python
Plug a few crasher in code module .
Merge with 3 . 4
bpo - 33642 : bpo - 33679 : Revise and condense blurbs ( GH - 7500 )
Remove bogus unittest from unittest . fork ( ) .
pstats - up analysis ( GH - 12249 )
bpo - 40836 : Fix wrong arguments for Opera Browser ( )
labs ( ) takes a long for an input .
BaseThreadedTestCase . setup ( ) : stop special - casing WindowsError .
Issue : Fix order of EventLoop . h
Merged revisions 71758 via svnmerge from
The strop module and test_strop . py believe replace ( ) with a 0 count
: Fix broken links in Doc / README . rst . Patch by Chris Jerdonek .
Make getsourcefile ( ) succeed even if the filename doesn', "'", 't end in', "'", '. py', "'", '- -
( py - continuation - offset ) : Update docstring to describe that this
Issue : Use specific asserts in http . cookiejar tests .
target libinstall :
Issue : Fix - up / build examples on clang
Don', "'", 't fail on Windows module
target libinstall :
issue8053 - logic was inverted on which platforms to run a framework directory .
Issue : Merge from 3 . 5
Merged revisions 77697 via svnmerge from
Issue : Fix test_warnings for locale encoding unable to encode
bpo - 35230 : Free buffer if realloc fails in tokenize . c ( GH - 12442 )
bpo - 30566 : Use _PyObject_ASSERT ( ) in typeobject . c ( GH - 10111 )
Constify filenames / iswide interoperability .
Hmm , __ne__ was missing
test_os : fix test on windows
The usual
clarifiy the documentation for issue17285 . see also commit
Mention the reader of unicode / data paths .
Merged revisions 83735 , 83939 , 83943 , 84166 - 84167 , 84257 , 85215 via svnmerge from
- Got rid of WITH_CYCLE_GC
( Merge 3 . 4 ) asyncio : document the debug mode
Issue : Fix faulthandler_thread ( ) of ceval . c to handle PyTuple_New ( ) failure
bpo - 29922 : Improve the error message in runpy when importing a module that includes the extension ( GH - 19239 )
Relocate the popen2 documentation since it is now substantially more
merge 3 . 3
bpo - 35250 : Correct SpooledTemporaryFile error message ( GH - 1034 )
Take out the setlocale ( ) call . It affects atof ( ) and will break
Merged revisions 76255 via svnmerge from
merge heads
Merged revisions 68221 via svnmerge from
bpo - 37126 : Fix includes and undefined server in Modules / tutorial . ( )
: fix typo in the sqlite module .
Issue : Merge 3 . 6 .
Merge 3 . 5
Remove sys_ * defines that are internal only and about to disappear
Don', "'", 't declare non - existant lines
Merged revisions 77809 via svnmerge from
Merge heads
test_math ( ) : In the extended calling syntax opcodes , you must check
bpo - 36341 : Fix gather to propagate cancel of itself with return_exceptions ( GH - 7209 )
: merge with 3 . 3 .
bpo - 28055 : Fix unaligned accesses in siphash24 ( ) . ( GH - 6123 )
Fix bad typo in the docstring about mkhowto .
: fix typo .
Issue : Expose the POLLRDHUP constant in the select module
Issue : Merge 3 . 4 .
Issue : Merge 3 . 5
Issue : Change cross - references in the tutorial .
Merged revisions 71540 , 71544 , 71546 , 71563 , 71572 , 71607 , 71653 via svnmerge from
Issue : Fixed test_regrtest for Windows
Issue : Fix PyErr_SetInterrupt ( ) documentation
Issue : Random generator objects now consume 2x less memory .
Merge with 3 . 4
This fixes issue 4332 for Py3k .
Cast bytes . from_bytes
Merged revisions 83818 via svnmerge from
Initial revision
Repair the Pstring ( ) call test for windows
Merged revisions 76774 via svnmerge from
Merged revisions 81838 via svnmerge from
bpo - 36965 : Port _overlapped module to multi - phase init ( GH - 22051 )
Merged revisions 84810 via svnmerge from
test_support .
Fixed a typo in the zlib test .
Merge 3 . 5 ( issue )
Mods by Just :
Fix typo .
Merged revisions 81880 via svnmerge from
Bump release level to 2 . 3 ( we won', "'", 't have time for this tomorrow ) .
argument - clinic : make multi - line string handling ( GH - 17852 )
Add a note that apply ( ) is needed since the extended call syntax is
Merge heads .
bpo - 30553 : Add status code 421 to http . HTTPStatus ( GH - 2589 )
The usual
- Issue : yArg_ParseTuple with " s " format when parsing argument with
Treat None as uid or gid .
Merge 3 . 4 ( odict )
Fix by Sjoerd changes
cpathname could be NULL if it was longer than MAXPATHLEN . Don', "'", 't try
Issue : Python PEP 498 .
Issue : Fixed raiseExceptions bugs in logging tests .
Repair inconsistent entry .
User trace API .
fix possible refleak in MAKE_FUNCTION ( closes )
Merge Py Idle changes :
test_math ( ) : In the extended calling syntax opcodes , you must check
Issue : Fix UnboundLocalError in AbstractDigestAuthHandler . get_algorithm_impls
Issue : Fix faulthandler_thread ( ) : Patch by Eryk
logging : fix wrong precedence link for 3 . 2 docs .
Merged revisions 75143 via svnmerge from
Issue : PyUnicode_FromFormatV ( ) now raises an error if the argument of
Moved include " Python . h " in front of other imports to silence a warning .
Fix part of SF bug # 544248 gcc warning in unicodeobject . c
Fix markup .
: fix typo in the docstrings .
Merged revisions 75634 via svnmerge from
Update link to the FIPS PUB 180 - 1 ( PDF is available , but not PostScript ) .
Fix waiter & input : applied to SF bug .
Merged revisions 81881 - 81882 via svnmerge from
Missing wishes
New string and URL .
bpo - 22490 : Remove __PYVENV_LAUNCHER__ from environment during launch ( GH - 9516 )
Strip IOErrors .
cleaned up distutils . command . build_py
Merged revisions 65152 via svnmerge from
Fix an inaccuracy in curses docstring .
Fix for issue 7852 : the DTD for OSX Plists has changed due
Merged revisions 80610 via svnmerge from
Add Amaury Forgeot d', "'", 'Arc .
Fixes for bug .
Documented the runtimemodel documentation .
Issue : Fix subprocess . Popen . import_ . test_packages to clean up
handle_system_exit ( ) flushs files to warranty the output order
bpo - 38858 : Add pycore_interp_init ( ) code to factorize code ( GH - 17483 )
Issue : Clarify the note that the coefficient of a Decimal
Update docstrings for the overview ( ) function .
Re - write the description of the os . spawn * ( ) functions , and cover the
asyncio : pep8 - ify the code .
Merged revisions 83818 via svnmerge from
Try to PyUnit .
Merge heads
Merge with 3 . 4
Merge heads
: merge with 3 . 3 .
Merge heads .
Merged revisions 85935 - 85936 via svnmerge from
Merge 3 . 4 ( test_urllib2net )
bpo - 36786 : Run compileall in parallel during " make install " ( GH - 13078 )
Initial revision
bpo - 40742 : Ensure that ms . key_compare is always initialized in list_sort_impl ( ) . ( GH - 8710 )
Merged revisions 70137 via svnmerge from
: fix typo in turtle . rst .
Merge 3 . 6 ( issue )
Issue : Fix Shift - JIS incremental encoder for error handlers different
Merge heads
Clean up some comments from MSVC
cosmetic cmp_lt ( ) : fix test handling of pickling ( ) .
Issue : Fix faulthandler_thread ( ) : release cancel lock before join lock
Issue : Update the note about the whatsnew .
Add', "'", 'get_platform ( )', "'", 'to construct a string that identifies the current
Use correct . module , not __builtins__ .
Fixes issue : Call NEWS support for PySequence_Tuple ( ) .
bpo - 26730 : Fix test_winconsole failures ( GH - 17400 )
Add a note about pow ( ) that it is no longer applicable .
Issue : The fromhex ( ) class methods of bytes and bytearray subclasses
Issue : The fromhex ( ) class methods of importing the
: fix typo in turtle documentation .
bpo - 31464 : Optimize _Py_hashtable_get_ptr ( ) ( GH - 20066 )
Combine the names to logging docs .
Add various items
Fix typo .
Typo fix .
Issue : Document NASM 2 . 10 + as requirement for building OpenSSL 1 . 0 . 1 on Windows
lint .
bpo - 29719 : Remove Date and Release field in whatsnew / 3 . 6 ( GH - 494 )
Issue : The binascii module now uses Argument Clinic .
sys . check_interval = x - > sys . setcheckinterval ( x )
bpo - 37253 : Fix typo in PyCompilerFlags doc ( GH - 14036 )
Mark linkage c by Laura Escoto replaces
Issue : Fix test_pdb to run zlib on Windows .
Issue : Fix incorrect precedence of signals in Decimal module .
Patch : list allowed states in error messages for imaplib .
cleaned up distutils . command . build
issue8053 - logic was inverted on which platforms to run a test on .
: fix man section
Merge
s / _alloca / alloca / g . h ( cs )
Add tests for the new . get ( ) and . setdefault ( ) methods of rfc822 . Message
Bumping to 2 . 6 . 6rc2 .
merge heads
Make decimal . ContextManager a private implementation detail of decimal . localcontext ( )
Issue : Merge from 3 . 5
_Py_wreadlink ( ) uses _Py_char2wchar ( ) to decode the result , to support
Added note for building sqlite and codepoint2name
H - 32827 : Fix usage of _PyUnicodeWriter_Prepare ( ) in decoding errors ( GH - 8505 )
Refined venv tests ( Ross Andrus )
Fixed defaultTestCase - > defaultTestResult
Prevent typos
Split the - fno - strict - aliasing .
Typo fix .
Backport 54644 :
: Fix typo in the howto .
Initial revision
bpo - 31454 : Include information about " import X as Y " in Modules tutorial ( GH - 4041 )
Merge with 3 . 4
Merged revisions 78136 via svnmerge from
Issue : Merge from 3 . 5
merge 3 . 3 ( )
Fix typo .
bpo - 41421 : Algebraic simplification for random . paretovariate ( ) ( GH - 21695 )
Merge with 3 . 5
: merge with 3 . 3 .
backport of sync with bare ampersands and pointy brackets ; this
backport media types from default
Look to the future : bump the version number .
: merge with 3 . 2 .
Rip out the HTML text into the language hierarchy .
Silence MSVC compiler warnings .
An interface to the font manager
Removed two unneeded lines from PyObject_Compare ( ) .
Merged revisions 77794 via svnmerge from
Fix markup of display links in stdtypes ( ) .
stop messing :
Issue : Fix a regression introduced by r78946 in the same .
Issue : Make the various ways for specifing weights produce the same results .
removed bizarre construct .
Removing IOErrors .
Added notes about doco
Issue : Fix UnboundLocalError in AbstractDigestAuthHandler . get_algorithm_impls
Initial revision
Merge 3 . 5 ( issue )
Issue : Merge from 3 . 5
Fix markup .
bpo - 35250 : Correct documentation for logging . contentmanager . raw_data_manager ( )
Issue : remove unneeded reload that allowed bad code .
Merge with 3 . 4
Issue : Tweak PEP 487 documentation
Test for 2 . x - pre1 . py .
Test sys . stdin version .
merge 3 . 5
Added . hgeol file and fix newlines in the 2 . 5 branch .
Added names for PyComplex_AsCComplex
merge 3 . 5 ( )
Merge
Fix issue11082 - Reject str for POST data with a TypeError . Document the need to explicitly encode to bytes when using urlencode .
Reindented test_socketserver .
Merge heads .
One more inittab .
correct len signature in docstring ( closes )
Typo fix .
Merged revisions 81275 via svnmerge from
bpo - 33666 : Add a toreadonly ( ) method to memoryviews .
Fix issue 26287 : While handling FORMAT_VALUE opcode , the top of stack was being corrupted if an error occurred in PyObject_Format ( ) .
Merged revisions 88481 via svnmerge from
sys . check_interval = x - > sys . setcheckinterval ( x )
: fix _dummy_thread import example .
Merge : merge with 3 . 5
bpo - 36786 : Truncate the column name after', "'", '[', "'", 'only if PARSE_COLNAMES is set . ( GH - 18942 )
merge
Optimize call to close mode .
Issue : Fix typo in AbstractDigestAuthHandler . get_algorithm_impls
Issue : Fix faulthandler_thread ( ) : release cancel lock before join lock
Issue : Fix Tools / comparecodecs . py . Patch by Serhiy Storchaka .
Merged revisions 70167 via svnmerge from
Merged revisions 84050 via svnmerge from
Enable TerminateProcess support to _subprocess driver
Merged revisions 77794 via svnmerge from
Add execve ( ) to factorize the code
Take out out - generated warning
Use os . closerange ( ) to find files to delete .
Issue : test_faulthandler is more tolerant on inaccurate time
Fixed the - - with - cxx - main build .
Merge with 3 . 4
Merged revisions 82780 via svnmerge from
Merge 3 . 4 ( test_urllib2net )
Moshe Zadka < mzadka @ geocities . com > :
Merge with 3 . 4
: ZipFile . join ( ) now differenciates buildir and srcdir
Issue : Compilation fix for recent AIX releases . Patch by David Edelsohn .
Remove outdated file
* wdbframewin . py ( ) : Initialize __privileged__ in globals so private
Issue : Use test_both ( ) in ceval . c .
Merge heads
Issue : Improved robustness of rotating file handlers .
Fixed defaultTestCase - > defaultTestResult
bpo - 32030 : Add _PyPathConfig_ComputeArgv0 ( ) ( )
Merged revisions 84408 - 84409 via svnmerge from
rephrase
Merge 3 . 5
Inline cmp_lt ( ) .
bpo - 36454 : Remove unused pyhash . h include from pythoncore . vcxproj ( GH - 19725 )
Merge 3 . 4 ( test_warnings )
merge 3 . 2
Patch : Add tp_getiter .
Fix indentation from Python3 . 4 to 2 . 6 .
do not expose break reference
bpo - 27737 : Allow whitespace only headers encoding ( )
merge 3 . 4 ( )
Issue : Fix faulthandler_thread ( ) : release cancel lock before join lock
bpo - 32947 : Do not swallow out of C API . ( GH - 6931 )
Change a comment in the distutils module .
Port 3 . 8 . 0a4
Apply Nicolas Riley', "'", 's Patch .
Issue : Merge 3 . 6 .
Merged revisions 68381 via svnmerge from
Fix issue6869 : refcount problem in the _ctypes module .
Issue : Add workaround for log1p ( ) in debug build .
enable refactor tests
Issue : Cleanup unicode_writer_prepare ( )
Issue : Merge test_gdb fix from 3 . 5
Merge 3 . 5
Clarify update .
AddrlistClass . getdomainliteral ( ) : rfc822 requires that the domain
The usual
Remove unused import
bpo - 40275 : Use new test . support helper submodules in tests ( GH - 21743 )
Update to MvL', "'", 's patch to avoid returning 2 when tp_compare
Make sure filter ( ) never returns tuple , str or unicode
More change for Makefile .
sprinkle const
Issue : Allow ValueError when marshalling too large object ( a sequence
Merged revisions 68116 - 60244 via svnmerge from
Merged revisions 78322 via svnmerge from
: slight clarification .
bpo - 33989 : Port _signal module to multi - phase init ( GH - 22051 )
bpo - 33383 : Make sure decimal context doesn', "'", 't affect other tests . ( )
bpo - 42748 : test_asdl_parser now uses exec_module instead of load_module ( )
Fix markup .
Continue rolling back pep - 3141 changes that changed behavior from 2 . 5 . This
pstats ( ) : New helper : run a file that takes a file descriptor
: fix test suite due to fix .
Issue : Expand expensive PyUnicode_READ ( ) macro in unicode_compare ( ) :
Merged revisions 86940 via svnmerge from
Merged revisions 71540 , 71544 , 71546 , 71563 , 71572 , 71607 , 71653 via svnmerge from
va_list is defined in stdarg . h .
bpo - 9216 : Nobody expects the geohashing FIPS inquisition ( GH - 19520 )
Merged revisions 79858 via svnmerge from
Fix typo .
bpo - 28298 : make array', "'", 'Q', "'", ',', "'", 'L', "'", 'and', "'", 'I', "'", 'accept big intables as elements ( )
Workaround for issue 1695 .
merge 3 . 3
Merged revisions 68306 - 68308 , 68340 , 68368 , 68422 via svnmerge from
merge 3 . 4 ( )
Re - word
merge heads
bpo - 38449 : Fix test_master_read ( ) ( GH - 728 )
start numbering things 1 . 5b1 now
Merged revisions 80404 via svnmerge from
Added tests for weakref module .
Issue : Fix installation of lib2to3 grammar pickles to ensure
Merge with 3 . 2
Added WNOHANG
: Refactor permissions of mbox , MMDF and Babyl mailbox files on flush ( )
merge with 3 . 2
small GUI updates to zlib .
Use PyModule_AddIntConstant ( ) instead of Tcl_SetResult .
Mention the UTF - 16 encoding speedup in the whatsnew ( issue ) .
merge with 3 . 3
: Fix typo in the documentation .
Merged revisions 85678 via svnmerge from
pyexpat code cleanup and minor refactorings :
tag 2 . 7 alpha 1
ignore control / l . h left - up .
Various edits
Issue : Fix Python 2 . 6 with patchlevel . OpenSSL , increasing both
Fix typo .
bpo - 33041 : Fix downcast warning on Windows ( )
Merge from 3 . 5
Issue : Merge rlcompleter change from 3 . 4 into 3 . 5
Issue : Get rid of the refcounting hack and speed up urllib . unquote ( ) .
Fixed an error in previous commit .
Fix markup .
PyTuple_New ( ) : vrbl i no longer referenced , so removed it ( which kills
: merge with 3 . 3 .
Merged revisions 81820 via svnmerge from
Merge issue : Fix collapsing of code sample in tutorial .
Fix mismerges of version directives .
 merge with 3 . 1 .
Use FSpCreateResFile ( ) in stead of CountMItems ( ) .
Merge heads .
Merge doc fixup from 3 . 5
typed_subpart_iterator ( ) : Add a brief " the " format
import . c does neither need mode_t nor _mkdir ( ) anymore
Commit this so CVS doesn', "'", 't keep noticing the next version number .
Try to fix test_gdb on s390x buildbots
Merged revisions 71280 via svnmerge from
Merge with 3 . 4
Update the __version__
M . - A . Lemburg < mal @ lemburg . com > :
Merged revisions 82169 via svnmerge from
Address some warning
Issue : Give informative error message when hhc . exe cannot be found .
merge heads
Merged revisions 86281 , 86285 via svnmerge from
Added pgrep ( ) routine , which is like grep / egrep / emgrep but uses Perl
Merge with 3 . 3
- confusing markup and comment in docstring
Merged revisions 83442 via svnmerge from
Issue : Add with revisions
Issue : Fix test_regrtest for installed Python
Total rewrite
Merge 3 . 4 ( test_warnings )
merge with 3 . 2
bpo - 33807 : Fix test_time . test_monotonic ( ) ( GH - 12929 )
: merge with 3 . 3 .
Use floor division where a class .
Issue : Add configure for textView .
Issue 10787 : Document the probability density function for random . gammavariate .
minor edits
Issue : Fix call_function ( ) of ceval . c to handle PyTuple_New ( ) failure
Patch : Add askopenfile slots to mimetypes . types_map
Adding assertItemsEqual with system . Added support for older expat versions .
merge with 3 . 3
Merged revisions 85400 via svnmerge from
Fix refleak .
Fix SF bug # 561858 Assertion with very long lists
merge with 3 . 3
: fix a bug .
Issue : Complete removal of implicit coercions for the complex
Merged revisions 78058 via svnmerge from
String method conversion .
Merged revisions 77798 via svnmerge from
Issue : Clarify kwargs field in sched . Event named tuple .
Fix an error in previous commit .
Update typing . py .
Issue : Raise ValueError when marshalling too large object ( a sequence
Issue : Don', "'", 't pass sys . stdout anymore
Issue : Add PyType_GetSlot .
: fix several factual errors in the docs .
Merged revisions 78058 via svnmerge from
Head merge
bpo - 38212 : Increase MP test_queue_feeder_donot_stop_onexc ( ) timeout ( GH - 16348 )
Issue : Add PyType_GetSlot .
do_cmd_rfc ( ) : Simplify the generated markup .
bpo - 39176 : Improve the error message in runpy when importing a module that includes the extension ( GH - 19239 )
Merge heads
Merge with 3 . 4
unicodeobject . c : Add MAX_MAXCHAR ( ) macro to factorize code handle
Merge : Fix wording with 3 . 3 .
Add items
Merge 3 . 5
bpo - 40401 : PyEval_SetTrace ( ) logs unraisable error ( GH - 18977 )
PyBuffer_New ( ) : Apply the change suggested in SF bug which
Boost link
Merged revisions 85400 via svnmerge from
va_list is no longer used
Added assert resword .
Issue : Fix deprecation warning in audio tests .
bpo - 33499 : Fix pymain_init_pycache_prefix ( ) ( GH - 8596 )
Issue : Improve bytes docs .
Merged revisions 81179 via svnmerge from
Issue : Don', "'", 't pass the Global
New demo sections .
bpo - 31454 : Include the', ['PEP section
Patch : FileInput now can be given an opening hook which can
Issue : Add log2 function to math module . Patch written by Mark
Issue : Fix faulthandler_thread ( ) of zlibmodule . c to handle PyTuple_New ( ) failure
Issue : reindent . gettempdir ( ) and tempfile . mkdtemp ( ) now try again
bpo - 30636 : Fix the indentation for the help ( )
Added descrobject . c and mysnprintf . c
Merged revisions 68306 - 68308 , 68340 , 68368 , 68422 via svnmerge from
: merge with 3 . 2 .
Change PyOS_Readline declaration to match the recent change to myreadline . c
Relocate the popen2 documentation since it is now substantially more
Merge with 3 . 5
Whitespace normalization .
Merge with 3 . 4
Merged revisions 74704 via svnmerge from
Fix markup .
Issue : Corrected fileConfig ( ) documentation .
Don', "'", 't mix < ioctl . h > unless on __VMS .
Merge heads
Update for math stubs
Whitespace normalization .
merge 3 . 3
bpo - 32468 : Improve frame repr ( ) ( )
Issue : On Windows , increase the stack size from 2 MB to 4 . 2 MB to fix
add a test for ; the exception must be normalized for with
merge
Merge
1 . = = x !
Mostly fix .
Merged revisions 80404 via svnmerge from
Fix variants of max_fd for r87454 .
Fix some problems .
asyncio : pep8 - ify the code .
Merge 3 . 3
Issue : TarFile . extractfile ( ) failed to resolve symbolic links when
Add missing comma .
Merge with 3 . 4
: fix typo .
Merged revisions 83818 via svnmerge from
bpo - 39889 : Fix test_master_read ( ) so that it succeeds on all platforms that either raise OSError or return b " " upon reading from master ( GH - 23536 )
Typo
Look to the future : bump the version number .
First crack at the Distutils " config " command . Unlike other commands , this
Merge 3 . 4
Clarify the " ARTICLE " - - > Send an " ARTICLE "
Issue : Random generator objects now consume 2x less memory on 64 - bit .
Move UserString to collections .
Merge tag', "'", 'v3 . 10 . 0a3', "'
Merged revisions 81214 , 82302 , 82465 , 83090 - 83091 , 84097 , 84099 via svnmerge from
Merge v3 . 4 . 2rc1 release stuff to default
Issue : Skip test_case_operation_overflow on computers with low memory .
New wishes
Issue : Fix uint_converter ( ) in zlibmodule . c , fix the " > UINT_MAX " check
Removed redundant inclusion of < errno . h > ( sorry , Roj ! ) .
Fixed an error in previous commit .
bpo - 36710 : Add PyInterpreterState . runtime field ( GH - 17270 )
Various few patch .
Merge
Update version 1 . 1 . 5
Fix prepare_ssl . py for OpenSSL 1 . 0 . 2a builds without Perl ( using old system ) .
Merged revisions 87317 via svnmerge from
document InvalidURL exception
Issue : Skip test_case_operation_overflow on computers with low memory .
bpo - 36302 : PYTHONPYCACHEPREFIX What', "'", 's New entry ( GH - 7749 )
bpo - 38377 : Add support for bytes paths in os . fwalk ( ) . ( GH - 6123 )
Merged revisions 81687 via svnmerge from
PyUnicode_EncodeUTF8 ( ) : tightened the memory asserts a bit , and at least
Merge 3 . 4 ( test_urllib2net )
Regenerated heads
replace _XOPEN_SOURCE on FreeBSD and running the test suite .
New file from sandbox and bgen
Initial revision
Issue : Fix a test_signal failure in the implementation of binary streams .
Fix docs for ABCs .
Merged revisions 85935 - 85936 via svnmerge from
Merge from 3 . 2 : 4
Expand the documentation of the low - level tracing / profiling interface .
Latest gramamr
Merge 3 . 2
Fix signature
Added getfiletype routine
Merged revisions 77389 via svnmerge from
Issue : Raise ValueError when marshalling too large object ( a sequence
Issue : Temporarily use www . google . com while investigating
Patch : Explain filter in terms of list comprehension . Remove
Issue : Clarify error message for do_richcompare ( )
Whitespace normalization .
Merged revisions 77877 via svnmerge from
What', "'", 's New in 3 . 3 : reorder modules
Merged revisions 74704 via svnmerge from
PEP 489 : Multi - phase extension module initialization
* xxobject . py : added tp_as_number , tp_as_sequence , tp_as_mapping to
Removed unused var
Markup fix
Remove outdated method from py options .
: fix refleak in the sqlite module .
Modified for the freeblock ( ) call .
Issue : Added tests for CLI of the calendar module .
[ Bug ] Fix misleading option text
Add news entry for 57f0af61da53 .
bpo - 38820 : OpenSSL 3 . 0 . 0 compatibility . ( GH - 17190 )
making BuildWinInstTestCase silent in case bdist_wininst is not run under win32
Added FOUR_CHAR_CODE define
Merged revisions 83818 via svnmerge from
Fast wishes
Document the proper exception to be raised by I / O operations on closed
Merged revisions 82492 via svnmerge from
bpo - 30317 , test_multiprocessing : fix test_timeout ( ) ( GH - 7957 )
IDLE refactoring : add gdb - topics
merge heads
Merged revisions 73880 via svnmerge from
Patch by Philip Jenvey :
Import bdist_msi
Issue : Get rid of the refcounting hack and speed up urllib . unquote ( ) .
Improve the description of difflib .
Issue : Improve cross - references in builtin types and functions documentation .
Issue : Don', "'", 't pass str ( _charset ) to MIMEText . set_payload ( )
code_repr ( ) , com_addop_varname ( ) , com_list_comprehension ( ) ,
SF bug 601775 - some int results that should be bool .
Various edits
merge with 3 . 2
remove requires_ssl decorator
: fix typo .
- Issue : Fix _PYTHON_PROJECT_BASE for srcdir ! = builddir builds ;
bpo - 32827 : Fix _Py_InitializeEx_Private ( ) ( )
Check return result for errors
Issue : Fix faulthandler_thread ( ) to handle correctly new_keys_object ( ) failure
add a new file for file Scan .
Merged revisions 86281 , 73872 via svnmerge from
Fix issue introduced by Jack ( 2006 - 08 - 1000110 )
Try to fix issue : skip the chances of the complex
math . name is pointless
Issue : Merge from 3 . 5
Whitespace normalization .
Issue : Fix a crash in str . decode ( ) when error handler returned
Revert newind . py checkin .
Issue : Fix re - recursive French code error .
Add pgrep ( ) example to factorize code .
Merged revisions 82169 via svnmerge from
new section - in constants
Add link to xdrlib module in " See also " section .
merge with 3 . 2
Fix use of exceptions .
Issue : Check for ERROR_ACCESS_DENIED in urllib2_localnet tests
start numbering things 1 . 5b1 now
Merge 3 . 4 ( logging )
Patch 1352 ( continued in issue 1329 ) by Christian Heimes .
bpo - 15248 : Emit a compiler warning when missed a comma before tuple or list . ( GH - 11757 )
update two stale documentation from doc example .
Fix the roundtripping function .
Fix refleak in error message
* wdbframewin . py ( re_eval ) : set __privileged__ in globals so private
Merged revisions 76774 via svnmerge from
Issue : handle PyUnicode_Writer ( ) error
Issue : handle PyUnicode_Writer ( ) error
Merge 3 . 4 ( asyncio )
Merge with 3 . 4
count ( ) should return integers
Add links to the math module .
Latest gramamr
: fix typo .
Fixed another clarification in the docstring .
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
Have use tempfile . h
Issue : Merge from 3 . 5
Fix by Eric Raymond : make the code that looks for various bits of
bpo - 34913 : Add _PyPreConfig . _config_init ( GH - 13481 )
bpo - 30780 : Add PyInterpreterState . _config_init ( GH - 13481 )
added lots of mactcp labels
issue8053 - logic was inverted on which platforms to run a test on .
Added a note about the extern ugliness .
Issue : Skip test_case_operation_overflow on computers with low memory .
ensure no one tries to hash things before the random seed is found
Issue : Factor out the _SuppressCoreFiles context manager into test . support .
- Remove if - untrue - f - run option
Added \\ exindex to \\ obindex
Issue : Improved robustness of rotating file handlers .
all fix
Benjamin Peterson writes :
Fix test_metaclass
bpo - 31191 : Add what', "'", 's new entry for os . errno removal ( GH - )
Merged revisions 72260 via svnmerge from
Fix a comment and it was inadvertantly in my library .
Add more items
Variant of SF patch 423181
bpo - 41729 : Fix test_winconsole failures ( 3 ) and hang ( GH - 22146 )
Feeble cleanup
Bgen now understands an ValueError ( closes )
Merged revisions 77389 via svnmerge from
Issue : Handled the guts of Queue to be more understandable and correct .
closes bpo - 39261 : Remove dead assignment from pyinit_config . ( GH - 17907 )
add TkVersion , TclVersion , don', "'", 't drop in debugger
Merged revisions 85400 via svnmerge from
whatsnew / 3 . 5 : Add missing word " class "
added US - topics
Issue : Rename initialisation 2 . x support
Issue : Add Python version to Idle editor window title bar .
Make decimal . ContextManager a private implementation detail of decimal . localcontext ( )
use isinstance
Merge heads
* Python 3 . 5
Issue : Add a toreadonly
Issue : TextIOWrapper . join ( ) now waits for the underlying thread state to be destroyed before returning .
Merge
Fix for test_multiprocessing
Merged revisions 83442 via svnmerge from
Issue : Fix FileIO . readall ( ) on Windows for files larger than 2 GB
1 ) List packages
( Merge 3 . 4 ) asyncio : document the debug mode
merge 3 . 5
bpo - 33989 : Fix test for Opera . c ( GH - 9451 )
Add the calendar synopsis .
fix refleak in the error case
Duncan Grisby noted a typo in _DummyThread .
Issue : Merge from 3 . 5
err is no longer used
Issue : The binascii module now uses Argument Clinic .
Issue : Add _Py_read ( ) and _Py_write ( ) functions to factorize code handle
Add a test for the HeaderParser class .
Issue : fix link
PyUnicode_EncodeUTF8 ( ) : tightened the memory asserts a bit , and at least
Issue : Merge from 3 . 5
Don', "'", 't try to create the directory if it already exists , otherwise the test fails
Issue : Add a new PyFrame_FastToLocalsWithError ( ) function to handle
update
Fred Gansevles < gansevle @ cs . utwente . nl > :
deprecate use of FakeSocket
Merged revisions 85935 - 85936 via svnmerge from
Merge 3 . 5 ( odict )
Changes for bug 4184
: fix sporadic warning in TestMaildir . test_create_tmp
- EditorWindow . parser ( ) now differenciates buildir and srcdir
: fix typo .
Add example
use macros
Merge socket doc changes from 3 . 3
Merged revisions 68306 - 68308 , 68340 , 68368 , 68422 via svnmerge from
Fix typo in comment leading up to _PyImport_FixupExtension ( ) .
Fixed distutils tests to use unittest . main ( ) for test discovery
Issue : Use test_both ( ) consistently in unicode_compare ( ) :
# merge r68096 , 3 . 3
Merged revisions 81687 via svnmerge from
Fix Issue10793 - hashlib documentation issue on return type of digest
Updated SysLogHandler documentation .
Logical up docstrings .
: slight clarification .
Issue : Add function test for _thread .
Fix test case for issue .
: fix typo in init . rst .
point from the iSilo conversion of 2 . 0 .
Moved macfsn / iswide packages to silence compiler warning
Merge Py Idle changes :
Added version notes .
Bump version to 3 . 2 . 2rc1 .
Fix typo .
bpo - 34043 : Optimize collections . Counter ( ) documentation ( GH - 5185 )
Fix markup typo
Added tag v3 . 3 . 0 for changeset de6b91d97113
Fix the ambiguity in the builtin install now contain only - use ,
Allow alloca .
PyTuple_New ( ) : vrbl i no longer referenced , so removed it ( which kills
Issue : Remove redundant checks for RuntimeError
bpo - 41756 : Add status code 421 to http . HTTPStatus ( GH - 2589 )
Two small fixes .
Merged revisions 75887 via svnmerge from
condense sentence
Issue : PyType_Ready ( ) now checks that tp_name is not NULL .
Merged revisions 79866 - 79867 via svnmerge from
Issue : Support WindowsRegistryFinder
Issue : xxlimited . c errors when building 32 and 64 bit on Windows
Merge 3 . 5
Fix test_poll . rst . By Christian Heimes .
Disable pythons own eventhandling .
Use isinstance
Issue : Fix a crash in the TextIOWrapper . tell method on 64 - bit platforms .
Update some URLs
Merge Py releasing changes :
bpo - 24758 : Improve the error msg for each __classcell__ . ( GH - 6931 )
Add missing comma .
Merged revisions 82769 via svnmerge from
In the multifile . py file as per security .
Cast None and reformat
bpo - 32362 : Fix references to non - existent multiprocessing . Connection ( ) ( GH - 6223 )
Variant of patch : don', "'", 't exit the interpreter on a SystemExit
count ( ) should return integers
email is at version 2 . 5 now
Merge with 3 . 4
Proof 3c9512d8ac0d
Merge with 3 . 2
SRE Nicolas
builtin_apply ( ) : Second argument type check is relaxed to allow any sequence .
Issue : The ssl module now has support for server - side SNI , thanks to a : meth : ` SSLContext . set_servername_callback ` method .
Issue : Fix a couple of places where undefined behaviour can
: fix the regression in the graphlib module
parser__pickler ( ) : Apply the change suggested in SF bug which
Copyright fix
Issue : Add a new PyFrame_FastToLocalsWithError ( ) function to handle
bpo - 33209 : End framing at the end of C implementation of pickle . Pickler . dump ( ) . ( GH - 6363 )
: fix typo .
Clarify grammar
Silence a warnings that was used by accident , so simplify
Merged revisions 81881 - 81882 via svnmerge from
Removed reliance on gcc / C99 extension .
Add missing sentinel to PyCursesWindow_getsets
Patch : Disambiguate . get_versions ( ) doesn', "'", 't follow confused
Clean up docstrings for unicode ( ) .
Issue : The modulefinder module now supports extended opcode arguments .
Merged revisions 79858 via svnmerge from
bpo - 33804 : Increase MP test_queue_feeder_donot_stop_onexc ( ) timeout ( GH - 16348 )
[ merge from 3 . 5 ] Prevent HTTPoxy attack ( CVE - 2016 - 1000110 )
Rip out the information about the PSA , since the fate of that is not
allow ctime ( ) , gmtime ( ) , and localtime ( ) to take None as equivalent to an omitted arg
merge v3 . 4 . 2rc1 release stuff to default
Merged revisions 76774 via svnmerge from
Explain the advantages of reversed .
Issue : The ssl module now has support for server - side SNI , thanks to a : meth : ` SSLContext . set_servername_callback ` method .
Issue : Use test_both ( ) consistently in test_importlib .
Issue : Fixed comment in grp . h .
Issue : Compilation fix for recent AIX releases . Patch by David Edelsohn .
bpo - 9216 : Nobody expects the geohashing FIPS inquisition ( GH - 19520 )
Merged revisions 79543 via svnmerge from
- PyType_Ready ( ) : Initialize the ob_type field to & PyType_Type if it', "'", 's
Merged revisions 70397 via svnmerge from
Reindented the tabnanny ( ) recipe .
Added \\ modulesynopsis to \\ obindex
closes bpo - 34868 : Fix constness of members of tok_state struct . ( GH - 18600 )
Add spawnv and spawnve functions for Win32 platforms .
- merge with 3 . 3 .
Issue : Merge from 3 . 5
making BuildWinInstTestCase silent in case bdist_wininst is not run under win32
- AppleEvent the - style option .
Issue : Fix uint_converter ( ) in zlibmodule . c , fix the " > UINT_MAX " check
merge 3 . 2
Issue : Merge 3 . 4 .
Minor markup : typos
: fix _dummy_thread import example .
Merge with 3 . 4
Issue : Fix SSL recv ( ) consistently in the _PyTime codec pattern
Fixed typo .
Issue : Merge from 3 . 5
bpo - 40401 : Remove duplicate pyhash . h include from pythoncore . vcxproj ( GH - 19725 )
Make test_base64 pass .
parser__pickler ( ) : Use Py_DECREF ( ) when reference is known to be non - NULL .
update pydoc - topics
Issue : Skip test_case_operation_overflow on computers with low memory .
Delete pythons libraries .
Add a note about pow ( x , y ) equalling x * * y ( the " * * " operator
Added " Internet Config " to index .
Merge 3 . 5
Issue : Update the documentation for ProcessPoolExecutor to note that it will not work in the interactive shell .
Try to fix .
Module up the contributor name .
Merge 3 . 5
Sf patch : unicode object leaks refcount on resizing
: fix a bug .
Remove sys_ * defines table .
Merged revisions 72905 via svnmerge from
Update URL .
Update the class example
Generalizations .
Checked .
Port the correction from Neale and issue
Issue : Use test_both ( ) consistently in test_importlib .
On SGI , we need to define _SGI_MP_SOURCE before including errno . h when
Shut up an occaisonal buildbot error due to test files being left around .
Fix typo in comment
Note that the base64 version
Update Misc / NEWS .
Socket handler closed prior to end of test .
Merged revisions 84289 via svnmerge from
Mention droped support for issue .
Merge 3 . 2
Issue : IDLE now uses sys . __stderr__
Merged revisions 75887 via svnmerge from
Make sure zip_path is enough
Try to fix issue : skip some tests on broken Ubuntu OpenSSL
Merged revisions 82492 via svnmerge from
Issue : Add log2 function to math module . Patch written by Mark
Fix test_fnmatch tests for Solaris .
closes bpo - 34581 : Conditionalize use of __pragma in Modules / socketmodule . c . ( )
Merged revisions 81859 via svnmerge from
eval_code2 ( ) : In the extended calling syntax opcodes , you must check
Phase two of package import . " import a . b . c " and all variants now do the
Issue : Dict reduces possibility of 2nd conflict in hash table .
Issue : Fix test_pdb to run in refleak mode ( - R ) . Patch by Xavier de Gaye .
bpo - 31843 : sqlite3 . connect ( ) now accepts PathLike objects as database name ( )
Merged revisions 66162 via svnmerge from
Added BaseHTTPServer to newopenfileobject
Merge 3 . 2
Merged revisions 75143 via svnmerge from
merge with 3 . 2
Issue : imp . new_module ( ) uses Unicode
Merge 3 . 4
bpo - 29453 : Fixed SpooledTemporaryFile error 10022 ( GH - 17400 )
Fix markup .
PyErr_Display ( ) : tightened the memory asserts a bit , and at least
web tree checker
bpo - 42748 : test_asdl_parser now uses exec_module instead of load_module ( )
Merged revisions 70137 via svnmerge from
Merge with 3 . 4
Merged revisions 79866 - 79867 via svnmerge from
NNTP . xgtitle ( ) description .
ord : provide better error messages
Added tests for math . py
merge with 3 . 2
Fix a typo in the comment
Merge 3 . 4 ( pystate )
Merged revisions 75143 via svnmerge from
ensure no one tries to hash things before the random seed is found
Fix issue6869 : refcount problem in the leak branch .
bpo - 1635741 : Port _overlapped module to multi - phase init ( GH - 22051 )
Adds precheck for console filename to fix Windows 7 .
describe that we can', "'", 't work
AddrlistClass , __ne__ was missing
Merged revisions 83966 via svnmerge from
Fix typo .
Kurt', "'", 's roundup id is kbk , use that in maintainers . rst instead of
Merge heads .
Many merge r68096 :
Forgot to add this file . CDROM device parameters .
Patch : Add a new suppress_crash_popup ( ) function to handle
Merged revisions 76774 via svnmerge from
merge with 3 . 2
Merge with 3 . 4
add Joe
: merge with 3 . 2 .
Support for frozen packages , matching the revamped Tools / freeze .
_dispatch ( ) : Use get_content_maintype ( ) and get_content_subtype ( ) to
Issue : logging doc math . cmath . assertSameElements to assertItemsEqual and changing behaviour
Fix typo
Add BaseHTTPServer to list of contents of chapter .
Bump distutils version number to 2 . 0c1
: merge with 3 . 3 .
bpo - 33725 : Initialize wsgiref', "'", 's SimpleServer as single - threaded ( GH - 12977 )
Fix for .
Issue : add explicit reference in asyncore docs .
Issue : Improve documentation for Handler . handleError .
Customary 2nd submit of this file after it', "'", 's been changed .
fix distutils tests on Windows ( ) .
Merge with 3 . 4
added getname , getdefault , getminmax functions
err is no longer used
Explain the advantages of reversed .
Removed THINK_C support .
Patch 1352 ( continued in issue 1329 ) by Christian Heimes .
Merged revisions 73931 via svnmerge from
Issue : Merge from 3 . 5
Interface to the Mac manager
Merge with 3 . 4
Try to fix symbol on s390x buildbots
Issue : close ( ) methods in multiple modules now are idempotent and more
Issue : Fix multiprocessing docs typo
Merged revisions 79161 via svnmerge from
merge with 3 . 4
when print ( ) gets unicode arguments , sep and end should be unicode by default
Patch 1352 ( continued in issue 1329 ) by Christian Heimes .
Issue : Remove redundant doc field from PyType_Spec .
merge with 3 . 2
Merged revisions 72306 via svnmerge from
Added', "'", 'get_platform ( )', "'", 'to construct a string that identifies the current
String method conversion .
web tree checker
Issue : Make sure that the coefficient of a Decimal
AddrlistClass . getdomainliteral ( ) : rfc822 requires that the domain
Check in the new functions and speed - up .
Add spawnv ( ) to factorize the code
bpo - 37571 : Remove extra space in ctypes docs ( GH14764 )
Use Py_ssize_t for field sizes and offsets .
Rename name to my_strndup to avoid conflict witth GNU libc .
Merged revisions 82780 via svnmerge from
builtin_apply ( ) : Second argument type check is relaxed to allow any sequence .
Merge with 3 . 3
merge 3 . 2
Fix test_fnmatch .
Changes by Rob Hooft for GMP 2 . 0 . 2
Fix test_pty on OSF / Tk 1 . 1 and 8 . 1 .
bpo - 37116 : Add PyInterpreterState . _config_init ( GH - 13481 )
What', "'", 's New in 3 . 3 : reorder modules
Make some names to PyObject_Size ( ) .
Merge from 3 . 5
Add dummy_threading .
Merged revisions 71540 , 71544 , 71546 , 71563 , 71572 , 71607 , 71653 via svnmerge from
: document that copyfile ( ) can raise Error .
Backport 54644 :
merge 3 . 5
Fix typo .
Keep below 80 chars .
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
Issue : Merge from 3 . 5
bpo - 31904 : Fix pymain_init_pycache_prefix ( ) ( )
Make sure the arr is clearer .
bpo - 30566 : Add _PyPathConfig_ComputeArgv0 ( ) ( )
Add items
Ignore the runtimemodel and summing links .
AddrlistClass - > completely ( ) .
Explicitely list the metadata attributes to show
Hack ndiff to new standard arithmetic .
Merge heads
Merged revisions 74675 via svnmerge from
Fix SF bug # 561858 Assertion with very long lists
Fix test_fnmatch .
Prevent crash on shutdown which can occur if we are finalizing
Merged revisions 77520 - 77521 via svnmerge from
Merged revisions 83442 via svnmerge from
First crack at the Distutils " config " command . Unlike other commands , this
Added posixpath , winsound .
Have the mailbox constant is built .
Issue : Fix a test_signal failure with - Wstrict - overflow .
bpo - 33679 : IDLE : Re - enable color configuration for code context ( GH - 7199 )
Issue : Factor out the _SuppressCoreFiles context manager into test . support .
resolve issue 2014
Merged revisions 75634 via svnmerge from
: fix with overzealous - sh rather than
Update the date .
Merged revisions 74843 via svnmerge from
( py - compute - region ) : Simplification .
: fix typo .
Merged revisions 76156 via svnmerge from
psuedo merge : : clarify when URLErrors are raised by urlopen .
Introduced underlining of menu labels
Issue : Add log2 function to math module . Patch written by Mark
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
Issue : Fix a test_posix of accessing signals
Use a real bool for a flag option .
Fix test_fnmatch .
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
Add Samuele Pedroni
Merged revisions 81809 via svnmerge from
Merge with 3 . 5
Issue : Clarify kwargs field in sched . Event named tuple .
remove unused assignment .
Merge changed fixes from 3 . 5
email . xgtitle ( ) description : Add a missing " this " section .
asyncio : pep8 - ify the stream .
bpo - 35214 : Initial clang MemorySanitizer support ( GH - 10479 )
no more threading . py into r60150 debugger
bpo - 29854 : Fix segfault in call_readline ( ) ( GH - 728 )
Added a autobuild - quick target that only rebuilds HTML .
( Merge 3 . 5 ) Issue : Fix a crash in the builtin function in TextIOWrapper . decode ( ) when
Issue : Fix for - Wunreachable - free exceptions in path_converter .
Issue : PyType_Ready ( ) now checks that tp_name is not NULL .
Issue : Merge from 3 . 5
bpo - 28055 : Add _PyPreConfig . _config_init ( GH - 13481 )
: fix typo in the sqlite module .
Merged revisions 62792 , 62799 - 62800 via svnmerge from
__GNUC__ is always initialized with exactly two parameters , so simplify
merge heads
Remove obsolete IMAP4_SSL_PORT
Use correct markup .
Patch 1352 ( continued in issue 1329 ) by Christian Heimes .
: fix refleak in the sqlite module .
merge 3 . 2
Use Py_UNICODE_WIDE ( ) instead of Tcl_SetResult .
merge
bug : Verify kwlist only the message for binary mode
Document the runtimemodel attribute .
Merged revisions 81838 via svnmerge from
: fix .
Issue : The ssl module now has support for server - side SNI , thanks to a : meth : ` SSLContext . set_servername_callback ` method .
Updated the command docstring from . slb
__GNUC__ does not imply gcc version is present , so just check for version ( closes )
Revert change f860b7a775c5
Enable _PyArg_ParseStack to _PyArg_ParseStackAndKeywords
Fix trivial test for raise issue .
Remove support for Visual Studio flag . This touches lots of files , since
Merge doc fixes from 3 . 5
Add documentation for curses . panel documentation .
new_arena ( ) : In the extended calling syntax opcodes , you must check
BaseThreadedTestCase . setup ( ) : stop special - casing WindowsError .
Moved PyMac_GetFullPath from macgetargv . c to macfsmodule . c so it loads
SF bug : fix typo
Fix test_mutants on dict views .
Merge .
Logical markup .
reuse wave . from_bytes
- general markup cleanup
Minor improvements to the convolve ( ) recipe ( GH - 24012 )
Merged revisions 86940 via svnmerge from
bpo - 36613 : synchronise _Py_hashtable_get_ptr ( ) in typeobject . c ( GH - 10111 )
Typo fix .
Hmm , __ne__ was missing
Backport 3 . 4
Input a4wide here
Issue : On Windows , increase the stack size from 2 MB to 4 . 2 MB to fix
Modified underlining of menu labels
Use PyModule_AddIntConstant ( ) in stead of CreateResFile ( ) .
: document that splitlines ( ) can raise Error .
Merged revisions 73931 via svnmerge from
Use INSTALL_PROGRAM for makesetup
Fix test_poll . CID an example to catch all an understandable object .
Issue : Fix test_gdb failures on KeysView and ItemsView .
Merge heads
Merged revisions 68381 via svnmerge from
Adding ensurepip , cmath and non - MSC root .
make importlib . py pass of the Python command .
Remove uneeded unneeded imports .
Fixed tests with Tcl / Tk < 8 . 5 ( closes ) .
do_cmd_key ( ) : Obsolete .
Added getfiletype routine
Issue : Fix site . fullmatch ( ) on Windows for files larger than 2 GB
: if a header has non - ascii unicode , default to CTE using utf - 8
Getting rid of PopUp menu .
Issue : Document kwargs field in sched . Event named tuple .
Issue18314 changeset 46393019b650
merge 3 . 2
Issue : Added tests for CLI of the zipfile module .
new_arena ( ) : In the extended calling syntax opcodes , you must check
Issue : Add workaround for log1p ( ) in debug build .
Another edits
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
remove the svn : executable property from files that don', "'", 't have shebang lines
give a saner file after xxmodule . py .
User const
Issue : Fix a test_signal failure : the delivery order is only defined for
Added PyExc_NotImplementedError
Fix typo .
Issue : Fix the urllib . request . Request . is_unverifiable deprecation documentation .
Issue 23261 : Some 3k sizeof tests fail .
Issue : Improved documentation for Handler . handleError .
Whitespace normalization .
bpo - 32037 : Fix test_master_read ( ) so that it succeeds on all platforms that either raise OSError or return b " " upon reading from master ( GH - 23536 )
Merged revisions 80627 via svnmerge from
Issue : Fix broken link in the General Python FAQ .
Remove outdated XXX comments .
Fix issue 26287 : While handling FORMAT_VALUE opcode , the top of stack was being corrupted if an error occurred in PyObject_Format ( ) .
Teach the peephole optimizer to fold simple constant expressions .
Issue : Use specific asserts in unittest . getaddrinfo ( ) .
merge with 3 . 2
Fix typo in functional module
issue28082 : clarified Filter entry .
String method conversion .
Issue : The asynchat . async_chat . handle_read ( ) method now ignores
Added disclaimer to the sleep ( ) function .
ord : provide better error messages
Issue : Fix incorrect precedence of signals in Decimal module .
Added BaseHTTPServer to xdrlib .
added test for Win32 packages for ARM32 ( GH - 12669 )
new_arena ( ) : In the extended calling syntax opcodes , you must check
merge with 3 . 2
I do not think we will ever have auto - indent at the prompt , that is for IPython and the like .
Issue : Some expensive the socket module to read on OpenSolaris .
Issue : Add curses . unget_wch ( ) function
Merged revisions 76975 via svnmerge from
merge with 3 . 3
Issue : Use surrogate - related macros
Ensure that the mailbox is closed to prevent problems on Windows with removing
[ Bug ] Wrong results .
handle_system_exit ( ) flushs files to warranty the output order
: typo .
Cast argument to set_panel_userptr to void * . Fixes bug .
Bug : Fixing SRE parser to handle', "'", '{ }', "'", 'as perl does , rather than
: document a way to escape metacharacters in glob / fnmatch .
merge 3 . 5 ( )
bpo - 25910 : Add _PyPreConfig . _config_init ( GH - 13481 )
Issue : Replace the PyUnicode_READ ( ) function in unicode_compare ( ) :
Merge 2 . 2 . 8rc tags into 2 . 7 .
Merge 3 . 3
: fix typo .
Write a section
Added new codec hp - roman8 submitted as patch [ 996067 ] hp - roman8 codec .
Merge heads
Issue : Fix rare of Python 2 . 6 compatibility .
_decimal is now 100x faster than the Python version in the pi benchmark .
bpo - 1635741 : Port _overlapped module to multi - phase init ( GH - 22051 )
Added support for rich Headers for hypot , and use the
bpo - 34373 : fix test_mktime and test_pthread_getcpuclickid tests on AIX ( GH - 8726 )
merge
Issue : Fix zlib decompressor handling of unused_data with multiple calls to decompress ( ) after EOF .
> ! = None :
Added a - - ciphers option to Lib / test / ssl_servers . py
Fix test_fnmatch .
Issue : Add configure . GNU ( ) functions to fix
bpo - 38731 : Fix function signature of quiet in docs ( GH - 17719 )
bpo - 35537 : PyEval_SetTrace ( ) logs unraisable error ( GH - 18977 )
C + + compile .
Revert revision 41940 : the test causes - uall to
condense wishes
Merge with 3 . 2
Improve Misc / NEWS . txt .
Issue : Add a new PyFrame_FastToLocalsWithError ( ) function to handle
Merged revisions 76464 via svnmerge from
merge 3 . 5 ( )
Merged revisions 82169 via svnmerge from
Lots of adjustments to deal with the document content now being stored
Add a docstring to say " use " and " nt . " error .
Fix a fix for test_multiprocessing
merge 3 . 5 ( )
: merge with 3 . 3 .
Issue : Remove unneeded words .
fix NEWS leak in Misc / NEWS .
Issue : Use assertWarns in test_asynchat .
Merged revisions 80179 via svnmerge from
bpo - 31191 : Add what', "'", 's new entry for os . errno removal ( GH - )
merge heads
Anthony Baxter < anthony @ interlink . com . au > :
Patch 1471925 - Weak linking support for OSX
Merge 3 . 4 ( test_urllib2net )
Remove unused variables from PyObject_Compare ( ) .
Fix SF bug # 561858 Assertion with very long lists
Issue : from_float ( ) : call the subclass', "'", '__new__ ( ) and __init__ ( ) .
Clarify - E docs .
merge 3 . 3
post release updates
Fix markup .
Merge 3 . 5
* Python / pythonrun . c : add string " Python " to fatal error message
merge with 3 . 2
Issue : Fix a crash in str . decode ( ) when error handler returned
Issue : Compilation fix for recent AIX releases . Patch by David Edelsohn .
fix refleak in the error case
bpo - 38731 : Add PyFrame_GetBack ( ) function ( GH - 19765 )
Merged revisions 79858 via svnmerge from
asyncio : sync with Tulip
Issue : Extract generated clinic code in Modules / _pickle . c to separate file .
ignore control box .
Merge : : Preemptively fix some other possible timing issues .
Issue : Fix test_warnings for locale encoding unable to encode
Merged revisions 61687 - 60244 via svnmerge from
Issue : IOBase . FileIO . FileIO ( ) now differenciates buildir and srcdir
Fixed broken IMAP4_SSL_PORT
: make is_attachment a method .
Fix inconsistent command - line option .
Split PyBytes_FromObject ( ) into subfunctions
Issue : Clarify fileConfig ( ) and RAND_pseudo_bytes ( ) failure
Fix typo .
Fix test_fnmatch .
Merge 3 . 4 ( test_urllib2net )
Added a autobuild - quick target .
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
[ Bug ] FloorDiv AST node type not listed
Fix whatsnew typo
bpo - 37126 : Reorder includes in xmltok . c to avoid redefinition of _POSIX_C_SOURCE ( GH - 16733 )
Various minor changes .
PyErr_NormalizeException ( ) : tightened the memory asserts a bit , and at least
Merged revisions 83325 , 71544 , 71546 , 71563 , 71572 , 71607 , 71653 via svnmerge from
Issue : Fix test_gdb failures .
merge heads
Fix typo in markup
Clarify the " Other " environment example .
Improve typos
Issue : On Windows , increase the stack size from 2 MB to 4 . 2 MB to fix
Merged revisions 80610 via svnmerge from
Issue : rsplit ( ) now checks that the underlying binary version
Use FSpCreateResFile ( ) instead of a AssertionError on the signal module a bit .
New file and zlib .
Issue : Add inspect . isabstract ( ) function to fix
bpo - 33400 : Add PyInterpreterState . _config_init ( GH - 13481 )
Merged revisions 81820 via svnmerge from
ord : fix HAVE_GETRANDOM_SYSCALL check
Eliminate unnecessary get_wrapped_ ( _TO ) opcodes .
: .
Merged revisions 74843 via svnmerge from
Add note about pimp and bundlebuilder to the Mac section .
Merge 3 . 5 ( issue )
( Merge 3 . 1 ) Issue : import __hello__ prints " Hello World ! " . Patch written
PyObject_Dir ( ) : Merge in __members__ and __methods__ too ( if they exist ,
Issue :', ['Issue : Use PyOS_string_to_double instead of PyOS_ascii_strtod in
Patch # : Add POP3 over SSL support .
Merge for issue
Backport of doc changes and fix typo
merge with 3 . 2
Issue : Fix occasional failure of bsddb / test / test_lock . py . Thanks
Merged revisions 71758 via svnmerge from
typo fix
Try to fix the breakage on 64 - bit platforms .
Add Samuele Pedroni
Improve test_asyncore .
SF bug :', ['Simplify push / pop clusters in ceval . c
lint .
* wdbframewin . py : fix long - standing bug .
Merge typo fix .
bpo - 42748 : test_asdl_parser now uses exec_module instead of load_module ( )
sprinkle const
merge from 3 . 2
: merge with 3 . 3 .
news for 6643
Merge 3 . 4 ( test_urllib2net )
Issue : Add a note to whatsnew / 3 . 5 . rst for nested functools . partial calls
typo .
bpo - 29912 : IDLE : Add tests for configdialog keys ( )
Added " Internet Config " to factorize page .
Merged revisions 67832 via svnmerge from
Merged revisions 76156 via svnmerge from
bpo - 33679 : Refactor xml . etree . ( )
Merged revisions 76577 via svnmerge from
err is no longer used
bpo - 28681 : Improve documentation for pstat module and profile . ( )
Issue : Fix exception incorrectly raised by xdrlib . Packer . pack_int when trying to pack a negative ( in - range ) integer .
PyErr_Display ( ) , PyErr_WriteUnraisable ( ) : Coverity found a cut - and - paste
Lots of minor changes .
Add a note about removing unwanted sys . path stuff
allow ctime ( ) , gmtime ( ) , and localtime ( ) to take None as equivalent to an omitted arg
Issue : Fix set operations on KeysView and ItemsView .
Issue : Added PyType_GetSlot .
Merge
Merged revisions 77798 via svnmerge from
Merged revisions 71758 via svnmerge from
take linkage def outside of WITH_THREAD conditional ( closes )
Get rid of mactcp cruft , added Mlte and CF modules .
Patch : Document the loads ( ) function function
istutils version
bpo - 37148 : Add IDLE configdialog tests for tkinter ( )
Issue : Under Windows , os . listdir ( ) didn', "'", 't release the Global
Issue : logging : fix bug in Unicode encoding fallback .
Clarified \\ 558544 , for logical addressing .
Added the new user / hashlib info .
Issue : distutils . gettempdir ( ) now differenciates buildir and srcdir
merge
merge
Issue : Update Windows for Windows Scan .
Merged revisions 85400 via svnmerge from
Kill more differences of PyList_SetItem ( ) and PyList_SET_ITEM ( ) .
asyncio : sync with Tulip
Issue : Use surrogate - in - optimizations in warning .
Issue : Fixed re . fullmatch ( ) of repeated single character pattern
Fix regression in new Python . h .
- Got rid of WITH_CYCLE_GC
Merged revisions 79161 via svnmerge from
Add a placeholder for the poplib module .
Issue : Skip tests for localhost in urllib2_localnet tests
Patch : cygwinccompiler . get_versions ( ) patch
Improve docstring .
NEWS item about pimp .
Explain the advantages of reversed . py
asyncio : Add __weakref__ slots to Handle and CoroWrapper . Upstream issue .
Support sizehint in _fileobject . readlines , as documented .
Merge for issue
: fix test for windows
Remove obsolete NEWS failure .
Merge typo fix .
bpo - 33725 : Initialize wsgiref', "'", 's SimpleServer as single - threaded ( GH - 12977 )
small minor updates .
removed bizarre theme .
Fix refleak in the word docstring .
Do not better error message for windows
Make sure we require rid of the child thing .
Proof read / editing of abc . Added table of collections . Hashable etc . to
Issue : Fix test_pdb on Windows
Issue : Use specific file again to stay out of compatible
Merge 3 . 5
Test for __sun instead of __sun__ , since SUNWspro only defines the latter ;
merge heads
Issue : doctest can now find doctests in functions and methods
Added bastion analogous for CWGUSI and codepoint2name
Somehow from Trent', "'", 's 101162 : Python 3 . 0 .
Add a placeholder for the HeaderParser module .
Issue : Use surrogate - related macros
Issue : Added a new PyFrame_FastToLocalsWithError ( ) function to handle
Fix typo .
Prep for Python 3 . 1 !
: Fix test warning .
Merged revisions 75224 via svnmerge from
Issue : Skip TIPC tests if marshalling is used .
use macros
: remove mention of the autobuild - dev target .
Merge with 3 . 4
merge heads
Support name so that _ctypes is not defined .
Merge heads
Patch : Add status code constants to httplib .
Merged revisions 86281 , 86285 via svnmerge from
Merge 3 . 5 ( odict )
fix typo in a comment .
unicodeobject . c : Add MAX_MAXCHAR ( ) macro to factorize code handle
Issue : Renamed WinFireFox to WinFirefox
merge 3 . 2
Issue : Improve markup docs .
Fix markup .
Merged revisions 84810 via svnmerge from
Added _ssl , TclVersion and getminmax functions
M . - A . Lemburg < mal @ lemburg . com > :
SRE 2 . 1b2 : increase the chances that the sre test works on other
Merged revisions 77798 via svnmerge from
Issue : Fix test_warnings for locale encoding unable to encode
Changes by Steen
The usual
Fix signature
: fix refleak in kqueue', "'", 's - OO symbols .
Issue : Allow WindowsRegistryFinder
Add missing decref
Issue : Fix a SystemError when accessing the ` ` cell_contents ` `
Obsolete no longer parser .
Backport 54644 :
add many APIs for bool
Added PyMac_Initialize ( ) routine to factorize icons warnings
Merge : Fix wording from 3 . 2 .
bpo - 36618 : Add PyFrame_GetBack ( ) function ( GH - 19765 )
Issue : Add socket . inf . check__all__ ( ) to datetime . __all__
Issue : Negative ints keys in unpickler . memo dict no longer cause a
Document the DISTUTILS_DEBUG variable for gzip module .
Fix typo .
Typo fix .
Update version of xmlcore libraries .
Make the stdlib library gets destroyed
Add contributor info to the FIPS PUB 180 - 1 ( PDF is available , but not PostScript ) .
Merge heads .
bpo - 33666 : Add a toreadonly ( ) method to memoryviews . ( GH - 6466 )
Initial revision
Optimization for communicate ( ) : If only one of stdin / stdout / stderr is
Merged revisions 73824 , 78887 , 78895 , 78900 , 79024 via svnmerge from
Merged revisions 79316 via svnmerge from
Fix patch with somtimes uninitialized port
Issue : Rewrite the pickle extension to Argument Clinic .
Bug : don', "'", 't crash on Cygwin , as chmod doesn', "'", 't work reliably there
SF : Tutorial for r87454 .
Clarify - E update for 3 . x
: add versionadded tags for functions copied from cgi .
SRE Peters :
: merge with 3 . 3 .
PEP L :
Issue : Cleanup test_asyncore . Patch by Phil Webster .
Issue : Improve the documentation of the unittest module . Patch by Chris Jerdonek .
Issue : Add configure . exe to mimetypes . types_map
Merged revisions 80610 via svnmerge from
Add entry for curses . panel documentation .
Generalizations .
bpo - 36974 : Unify implicit and explicit state in Argument Clinic ( GH - 21426 )
Fix typo in comment ( closes )
Added " Internet Config " to index .
SF bug 601775 - some int API for OSX
Merge from 3 . 5 for issue
Merged revisions 80179 via svnmerge from
Fix two error in previous commit .
bpo - 28055 : Fix unaligned accesses in siphash24 ( ) . ( GH - 6123 )
merge heads
ord : provide better error messages
Check return results for errors
( py - compute - emacs - hook ) : Simplification .
Misc / NEWS entries for zipapp changes
bpo - 25910 : Delay removal of ABCs from collections . ( GH - 13409 )
( py - kill - emacs - hook ) : Simplification .
bpo - 30808 : Use _Py_atomic API for concurrency - sensitive signal state ( )
Note that abspath ( ) was added in 1 . 5 . 2 .
py - cvs - rel2_1 ( Rev 1 . 4 ) merge - move " import * " to module level
Issue : xxlimited . c errors when building 32 and 64 bit on Windows
Only call sq_repeat if the object does not have a nb_multiply slot . One
open ( ) gets bytes or error on Windows .
Issue : test_faulthandler reduces unencodable characters
Issue : Deprecate WindowsRegistryFinder
Merged revisions 77520 - 60244 via svnmerge from
cleaned up distutils . command . build_py
Merge 3 . 5 ( issue )
Fix typo
Merged revisions 81809 via svnmerge from
Mostly fixes .
Issue : Fix test_warnings for locale encoding unable to encode
Sigh . Strengthen the resriction of the previous checkin : tp_new is
bpo - 27788 : Nobody expects the geohashing FIPS inquisition ( GH - 19520 )
asyncio : sync with Tulip
bpo - 5028 : Fix gather to propagate cancel of itself with return_exceptions ( GH - 7209 )
: fix typo .
Issue : add explicit reference in asyncore docs .
Bug : Fixing SRE parser to handle', "'", '{ }', "'", 'as perl does , rather than
Worked around bug SF : changed the text of three sections that
err is no longer used
Merged revisions 84289 via svnmerge from
_Py_wreadlink ( ) uses _Py_char2wchar ( ) to decode the result , to support
Add tests for SF bug ( multiple inheritance ) .
Merged revisions 62792 , 62799 - 62800 via svnmerge from
Merged revisions 83966 via svnmerge from
Merged revisions 66162 via svnmerge from
Fix versionchanged directive in datetime . rst
Add missing sentinel to PyCursesWindow_getsets
Issue : Allow the guts of functools . cookiejar . write .
micro changes
remove debugging turd from weakref_segfault ( ) .
Issue : Fix occasional error message
The usual
bpo - 15248 : Remove unused pyhash . h include from pythoncore . vcxproj ( GH - 19725 )
merge with 3 . 2
- Modules / Setup . dist : Update pyexpat
: merge with 3 . 3 .
- merge with 3 . 3
Various markup tweaks .
Merged revisions 81774 - 81775 via svnmerge from
Merged revisions 72781 via svnmerge from
: if a text has a null node node .
Added', "'", 'get_platform ( )', "'", 'to construct a string that identifies the current
Fix markup
Issue : Correct faulty configure checks so that os . chflags ( ) and
merge 3 . 3 ( )
Avoid more zombi loop .
Merged revisions 86081 via svnmerge from
Issue : Separate initialisation from setting when initializing
Fix the typos
Now that we don', "'", 't have AC_CHECK_LIB ( m , pow ) , the HAVE_LIBM symbol
Merged revisions 69582 - 69583 via svnmerge from
bpo - 34373 : fix test_mktime ( ) ( )
merge 3 . 4 ( )
Bump release level to 2 . 3 ( we won', "'", 't have time for this tomorrow ) .
Added HAVE_MKTIME
Patch : Make terminal become controlling in pty . fork ( ) .
Issue : Remove breaked ( ) check in Misc / ACKS .
Before not _PyType_Lookup ( ) with stdin . c .
Revert issue 2014
Check return results for errors
Nits .
Fix bug introduced by r68451
Merged revisions 67082 via svnmerge from
merge with 3 . 2
New heros .
Richard Mortier < rmm1002 @ geocities . com > :
Issue : Fixed test_regrtest for installed Python
Close : Permission bits are once again correctly copied from the source file to the cached bytecode file . Test by Eric Snow .
Issue : Merge from 3 . 5
Don', "'", 't try to build ctypes when configuration of libffi fails .
Added __members__ and PyErr_NewException ( ) calls .
Merge tag', "'", 'v3 . 10 . 0a3', "'
delete bogus file
Issue : Fix incorrect precedence of signals in Decimal module .
Eliminate unnecessary get_wrapped_ ( u ) long defines in struct module .
unicodeobject . c : Add MAX_MAXCHAR ( ) macro to factorize filenames
Note that \\ var { ppos } values are better .
Merged revisions 79161 via svnmerge from
Issue : Under Windows , os . listdir ( ) didn', "'", 't release the Global
Merged revisions 70137 via svnmerge from
Merged revisions 75224 via svnmerge from
use macros
Issue : Add direct test for _thread .
Merged revisions 85400 via svnmerge from
Fix a handful of memory entry . ( GH - 17696 )
Merge', ['Try .
Issue : Update htest , use mbcs ( ) instead of join ( ) .
Marc - Andre Lemburg < mal @ lemburg . com > :
Merge with 3 . 4
Remove a dummy that .
: fix typo in turtle . py .
Issue : Merge from 3 . 5
Merged revisions 81687 via svnmerge from
Make socket . libc_ver ( ) less slow
: fix typo in the datetime module .
merge with 3 . 2
bpo - 35808 : Retire pgen and use pgen2 to generate the parser ( GH - 11814 )
: merge with 3 . 2 .
bpo - 38820 : mention 3 . 0 . 0 compatibility . ( GH - 17190 )
Issue : Merge 3 . 4 .
Revert bug .
Added \\ modulesynopsis , \\ obindex
Added missing static option to rag_nextrange . h .
: .
bpo - 33725 : multiprocessing uses spawn by default on macOS ( GH - 13603 )
Fix ( bpo - 31596 ) ( )
Merge 3 . 2 ( issue )
Merged revisions 76975 via svnmerge from
Add a note about pow ( ) and
Issue : Fix validation of relative imports in parser module .
Merge heads
bpo - 35808 : Retire pgen and use pgen2 to generate the parser ( GH - 11814 )
Merge 3 . 2
: fix typo in turtle . py
cgi . FieldStorage . read_multi ignores Content - Length
cleaned up distutils . command . build_ext
Add PyExc_NotImplementedError
Issue : Factor out the _SuppressCoreFiles context manager into test . support .
Fix the', "'", 'ensure_', "'", 'so it was the', "'", 'have_run', "'", 'flag for the
Fix
: fix refleak in the sqlite module .
start numbering things :
bpo - 30854 : Fix asyncio PidfdChildWatcher on waitpid ( ) error ( GH - 17477 )
Decimal is updated , issue :
bpo - 35780 : Fix leaks in test_logging ( )
bpo - 29636 : Add PyInterpreterState . _config_init ( GH - 13481 )
Issue : Fixed possible crash in PyUnicode_AsDecodedObject ( ) in debug build .
Typo fix
More changes for 1 . 2a2 .
fix \\ NULL with following space and no { }
: fix typo in NEWS entry .
est for correct issues in test_httpservers
Merge with 3 . 4
bpo - 42748 : test_asdl_parser now uses exec_module instead of load_module ( )
Fix the failures on cygwin ( 2006 - 08 - 10 fixed the actual locking issue ) .
Merged revisions 79866 - 79867 via svnmerge from
Note that abspath ( ) was added in 1 . 5 . 2 .
News item for issue
bpo - 36124 : Add a toreadonly ( ) method to memoryviews . ( GH - 6466 )
Issue : Merge from 3 . 5
Issue : Fix FileIO . close ( ) on Windows for files larger than 2 GB
Fix markup .
merge 3 . 4 ( )
merge with 3 . 3
Constify PyBytes_FromObject ( ) into subfunctions
Issue : Remove outdated mention of RuntimeError
Add a note about pimp and bundlebuilder to the Mac guide .
Update profile - topics
Logical markup .
Issue : Random generator objects now consume 2x less memory on 64 - bit .
: fix typo in the NEWS file
Support name from sandbox , not Programs .
Merge 3 . 2
Merged revisions 72905 via svnmerge from
fix distutils . Counting ( ) from distutils . Message
Added err_badcall ( ) and planned new errors .
Move misplaced NEWS entry to the top of the file .
Merged revisions 83701 via svnmerge from
apply issue 2014
Fix - up option problems on clang
Samuel L . Bayer :
Fix typo in markup
merge 3 . 5
Merge heads
start numbering things 1 . 5b1 now
Merge 3 . 2
Merged revisions 83442 via svnmerge from
Add ref to toctree .
BaseThreadedTestCase . setup ( ) : stop special - casing WindowsError .
Issue : Temporarily use www . google . com while investigating
bpo - 40755 : Add Py_DEPRECATED to deprecated unicode APIs ( GH - 20878 )
Merge with 3 . 4
: merge with 3 . 3 .
Issue : Keep the traceback of original exception in _PyErr_ChainExceptions ( ) .
Issue : Moves NEWS entry under Library .
bpo - 38858 : Add init_set_builtins_open ( ) subfunction ( GH - 17346 )
Merge with 3 . 4
closes bpo - 34581 : Conditionalize use of __pragma in Modules / socketmodule . c . ( GH - 9067 )
skip test_itimer_virtual on NetBSD to prevent the test suite from hanging .
Fix prepare_ssl . py for OpenSSL 1 . 0 . 2a builds without Perl ( using old system ) .
Docstring fix
Enable a autobuild - quick target that only rebuilds HTML .
Merged revisions 80990 via svnmerge from
Issue : Fix a link to StreamReader docs .
Issue : Fix a crash in str . decode ( ) when error message returned
bpo - 33666 : Add what', "'", 's new entry for os . errno removal ( GH - )
PyUnicode_EncodeUTF8 ( ) : tightened the memory asserts a bit , and at least
document the requestline and close_connection attributes , use real booleans , and add tests ( closes )
Fix a loooong - up brainfart : a problem in recvfrom ( ) .
DG / UX thread patches ( Ross Andrus )
bpo - 33400 : Implement PEP 570 changes in the tutorial ( GH - 21340 )
Anthony Baxter < anthony @ interlink . com . au > :
Merge heads .
Closes : merged documentation fix from 3 . 2 .
closes bpo - 34868 : Improve error message for', "'", '_', "'", 'is combined with an invalid type specifier . ( GH - 9666 )
merge with 3 . 3
Patch : cygwinccompiler . close ( ) patch
Fix issue 26287 : While handling FORMAT_VALUE opcode , the top of stack was being corrupted if an error occurred in PyObject_Format ( ) .
bpo - 36965 : Fix includes in main . c ( GH - 19725 )
Added PyExc_Exception , TclVersion .
Gregor Hoffleit < mal @ lemburg . com > :
Bump the __version__
Issue : Add PyType_GetSlot .
OSX : getwindow ( ) parent method and realize ( ) child method .
Remove unused function .
Issue : Fix a leak in io . BufferedIOBase . readinto ( ) when raw
Kill remaining mentions of import_nt . c
Merge issue : Fix collapsing from 3 . 2 branch .
Issue : Merge from 3 . 5
Issue : Fix test_warnings for locale encoding unable to encode
ignore { CLEAN , DEAD , FORBIDDEN } BYTE .
Fix Issue10793 - hashlib documentation issue on return type of digest
merge 3 . 3
bpo - 35790 : OpenSSL 1 . 1 . 1 - pre1 / TLS 1 . 3 fixes ( )
Moved include " Python . h " in front of other imports to silence a warning .
Issue : Fix a test_signal failure introduced by r78946 .
explicitly cast away constness to silence compiler warning
Explain the advantages of reversed . Fixes .
Fix PyUnicode_CopyCharacters ( ) doc
Merged revisions 72306 via svnmerge from
SF bug : datetime . update does not correctly escape argument to / usr / bin / file
Moved the freeblock ( ) function .
Documentation typo
Issue : Add a note for the spwd module . Patch by Mark
Merged revisions 77794 via svnmerge from
Merge with 3 . 3
bpo - 37126 : Fix includes in main . c ( GH - 19725 )
Issue : Merge from 3 . 5
Remove outdated note about constraining of the bit shift right operand . ( )
Minor boolean additions on DL_DIRECTORY .
Merge Py Idle', "'", 's Rev 1 . 16 ( string methods )
Add a note that apply ( ) is needed since the extended call syntax is
Fix for test_multiprocessing
Issue : Fix occasional failure of bsddb / test / test_lock . py . Thanks
Issue : All global methods in the getfullargspec .
Fix typo in new metaclass docs
Merged revisions 85082 via svnmerge from
: add a note to audioop docs about signedness of sample formats .
Issue : Skip tests for localhost in urllib2_localnet tests
Updated for Python 2 . 5b1
Merge
Issue : Remove', ['Issue : multiprocessing . timeout ( ) when raw
Backport 54644 :
Issue : Cross - reference types . SimpleNamespace from the namedtuple docs
Merge with 3 . 4
Merged revisions 75634 via svnmerge from
bpo - 35808 : Retire pgen and use pgen2 to generate the parser ( GH - 11814 )
: document that copyfile ( ) can raise Error .
: Make linecache docs reflect that all files are treated the same .
Use INT_MAX / LONG_MAX instead of non - ANSI LONG_BIT
bpo - 39889 : Fix gather to propagate cancel of itself with return_exceptions ( GH - 7209 )
Issue : Add tests for the spwd module . Original patch by Vajrasky Kok .
issue8053 to int here
: remove Py_GetBuildNumber ( ) , which was removed in favor of Py_GetBuildInfo ( ) .
Misc NEWS by Steen
closes bpo - 34868 : Improve error message for', "'", '_', "'", 'is combined with an invalid type specifier . ( GH - 9666 )
: fix documentation for the email docs .
bpo - 37726 : Sort list of sources ( GH - 12341 )
merge heads
Don', "'", 't hide < ioctl . h > unless on __VMS .
asyncio : sync with Tulip
make test_base64 pass .
Merged revisions 72781 via svnmerge from
Issue : Factor out the _SuppressCoreFiles context manager into test . support .
Fix docstring of len ( ) .
Merged revisions 75634 via svnmerge from
Merged revisions 77697 via svnmerge from
Moved Mark SummerField', "'", 's io doc changes back to the trunk . ( He will edit here in the future . )
bpo - 41302 : Fix gather to propagate cancel of itself with return_exceptions ( GH - 7209 )
- Got rid of WITH_CYCLE_GC
Merged revisions 82806 via svnmerge from
- The usual
Merge : Fix typo
Merge heads .
Adding deprecated error messages after informative
Issue : An process should now be used by default for the
Issue : Merge from 3 . 5
Merged revisions 77794 via svnmerge from
Issue : Clarify lenient in socket . Popen docs .
Recompiled binaries after source changes .
BaseThreadedTestCase module is enough
Issue : Skip test for localhost in urllib2_localnet tests
Issue : python - m tokenize support has been added to tokenize .
Stylistic tweaks to the bz2 module to make it easier to transplant changes from
Fix for bug 4184
Merged revisions 79858 via svnmerge from
( Merge 3 . 2 ) Issue : Fix a crash in the builtin function super ( ) when
Issue : Optimize error if % proc ) is to avoid
Merge with 3 . 4
Patch : FileInput now can be given an opening hook which can
Patch : FileInput now can be given an opening hook which can
Merged revisions 79543 via svnmerge from
Issue : On Windows , increase the stack size from path_converter .
Issue : merge from 3 . 5
merge 3 . 5 ( )
Fix markup .
Simple support to PyUnit .
Merged revisions 70439 via svnmerge from
bpo - 32726 : Optimize _Py_hashtable_get_ptr ( ) ( GH - 7200 )
Merge with 3 . 4
Fix mismerges of version directives .
- Issue : Fix _PYTHON_PROJECT_BASE for srcdir ! = builddir builds ;
Move misplaced notes in beta1 news
PyBuffer_New ( ) : Raise ValueError if size is negative ( the other
Merged revisions 84810 via svnmerge from
: logging . architecture return path for classes , not self

Issue : CVE - 2013 - 1752 : Change use of readline in imaplib module to limit
Issue : TextIOWrapper . extractfile ( ) accepts bytes , bytearray and str with
* xxobject . c : added tp_as_number , tp_as_sequence , tp_as_mapping to
Make reindent . py happy .
Merged revisions 81881 - 84409 via svnmerge from
bpo - 42748 : test_asdl_parser now uses exec_module instead of load_module ( )
\\ withsubitem additions .
lint now supports packages
Tagging for release of Python 2 . 5b1
Merged revisions 66539 via svnmerge from
bpo - 5028 : fix doc bug for tokenize ( GH - 11683 )
: fix wrong escape of the expat version in pindent .
Merged revisions 74675 via svnmerge from
Only call sq_repeat if the object does not have a nb_multiply slot . One
Initialize pythons addr declaration .
Markup fix
bpo - 42748 : test_asdl_parser now uses exec_module instead of load_module ( )
Regenerate with the new option for 32 - bit time_t
Fix issue 26287 : multiprocessing no longer leaks on windows
merge 3 . 4 ( )
Merged revisions 67694 via svnmerge from
Patch : FileInput now can be given an opening hook which can
Issue : Fixes broken link in Misc / NEWS
Typo fix
bpo - 42140 : Add IDLE configdialog tests ( )
bpo - 34519 : Add _PyPreConfig . _config_init ( GH - 13481 )
Issue : Remove uses of PyObject_IsTrue ( ) in pybench .
Fix PDB section in the zlib module .
cleaned out distutils to list options .
Issue : Python interpreter installer pass from 3 . 4 . 2 .
Issue : make it clear that create_default_context ( ) also enables hostname checking
Rip out the information about the PSA , since the fate of that is not
Fix typo
Fix typo .
Merged revisions 79866 - 79867 via svnmerge from
: add bullet point for __cmp__ removal .
Issue : imp . new_module ( ) uses Unicode
Merged revisions 87317 via svnmerge from
Backport 3 . 2
Merge from 3 . 2
Patch by kragen @ pobox . com : When tracing is turned on , lines shorter
New function .
Merged revisions 76577 via svnmerge from
Fixed issue11082 - Reject str for POST data with a TypeError . Document the need to explicitly encode to bytes when using urlencode .
Merge functional howto improvements with 3 . 3 .
merge heads
Always all the whitespace in the generated HTML .
asyncio : Fix ResourceWarning on Windows
removed unused var
Merged revisions 75356 via svnmerge from
Separate initialization from clearing .
Typo fix
Issue : Fixed build comments in Misc / ACKS .
: Credit Emanuel Barry in NEWS .
Issue : Fix the urllib . request . Request . is_unverifiable deprecation documentation .
Merge heads
bpo - 32374 : Free buffer if realloc fails in tokenize . c ( GH - 12442 )
Re - generated with - toolbox comments if Python', "'", 't buggy
bpo - 31080 : Allowed compileall . config . fileConfig ( ) to accept both args and kwargs . ( GH - 2979 )
bpo - 34303 : Nobody expects the geohashing FIPS inquisition ( GH - 19520 )
Fix typo .
: remove Py_GetBuildNumber ( ) , which was removed in favor of Py_GetBuildInfo ( ) .
add security
Test for __sun instead of __sun__ , since SUNWspro only defines the latter ;
Merged revisions 82301 via svnmerge from
Add a test for the new / / operator too .

Merged revisions 74704 via svnmerge from
Add a note that apply ( ) is needed since the extended call syntax is
cleaned up distutils . command . build
Issue : Fix uint_converter ( ) in zlibmodule . c , fix the " > UINT_MAX " check
bpo - 31454 : Implement contextlib . AsyncExitStack . ( GH - 12132 )
Add a note about pow ( x , y ) equalling x * * y ( the " * * " operator
fix \\ NULL with following space and no { }
'", 'sum', "'", 'is no longer a good variable name . Use', "'", 'total', "'", 'instead .
: merge with 3 . 3 .
Merged revisions 81809 via svnmerge from
libpython . py : defer call to gdb . lookup_type (', "'", 'PyUnicodeObject', "'", ')
__GNUC__ does not imply gcc version is present , so just check for version ( closes )
builtin_apply ( ) : Second argument type check is relaxed to allow any sequence .
Fix a typo in the distutils module .
Issue : Compilation reduces possibility of 2nd conflict in hash table .
Add ossaudiodev ; fix typo
Merge 3 . 5
Use INSTALL_PROGRAM for makesetup
Whitespace normalization .
Typo fix .
NNTP . xgtitle ( ) description : Add a missing " ) " from the response value
Fix markup of an \\ indexiii - - > only two parameters were given .
Merged revisions 83818 via svnmerge from
merge 3 . 3
Issue : Fix warnings in AbstractDigestAuthHandler . get_algorithm_impls
Add Samuele Pedroni
merge heads
merge 3 . 3
Merge
User bytes of c_char_p and c_wchar_p types .
Note the advantages of reversed . pop ( ) and PyOS_setsig ( ) .
Add GHOP students .
merge
rephrase
Various edits
Don', "'", 't get unexpected errors in PyErr_WarnExplicitObject ( ) . ( )
* Add test for 85cb90f79cbf .
Issue : An exception in Tkinter callback no longer crashes the program
satisfy the tabnanny ( thanks to MH for noticing the problem )
Issue : Fix test_warnings for locale encoding unable to encode
Merge
post release updates
Catch IOErrors .
: remove Py_GetBuildNumber ( ) , which was removed in favor of Py_GetBuildInfo ( ) .
Issue : Merge from 3 . 5
Combine the two seealso sections .
Issue : Fix subprocess docs typo
Merged revisions 80610 via svnmerge from
Issue : Remove redundant property and unused variables .
Repair a typo .
Add doc example for bool
Changes by Sjoerd ,
build _csv extension module
Add RELEASE and markup
bpo - 37126 : Update compare_networks ( ) ( GH - 7200 )
Patch : Make askopenfile PEP controlling to PyDateTimeAPI
Merged revisions 75519 via svnmerge from
Take PyUnicode_TransformDecimalToASCII out of the limited API .
Issue : Fixed writing ZIP files that starts not from the start of the
Merge 3 . 5 ( issue )
Issue : Fix a stack corruption in ossaudiodev module when the FD is
Merge heads
: merge with 3 . 2 .
Added notes about the eventual disappearance of two - parameter bind ( ) ,
Issue : Fix test_gdb on KeysView with low memory .
Updated documentation for the new slice arguments for list . index ( ) .
bpo - 23722 : Improve a code for absent __classcell__ . ( GH - 6931 )
Issue : Moves NEWS entry under Library .
Integrate the stack size from a trailing disappearance .
Issue : The binascii module now uses Argument Clinic .
Issue : Expose WindowsRegistryFinder
Issue : PyType_Ready ( ) now uses the new Unicode API .
bpo - 34271 : Fix test_master_read ( ) so that it succeeds on all platforms that either raise OSError or return b " " upon reading from master ( GH - 23536 )
Merge doc fixup from 3 . 5
: merge with 3 . 3 .
: merge with 3 . 3 .
Issue : Fix writing ZIP files that starts not from the start of the
Fix a poorly worded error message
Issue : Fix faulthandler_thread ( ) of zlibmodule . c to handle PyTuple_New ( ) failure
target libinstall :
Issue : Clarify fileConfig ( ) documentation .
merge
bpo - 35683 : Add _PyPreConfig . _config_init ( GH - 13481 )
Issue : Extract generated clinic code in Modules / _pickle . c to separate file .
Update some typos
fix parse_syntax_error to string_join .
: fix refleak in the sqlite module .
rewrite nocaret test to unittest .
merge 3 . 4 ( )
bpo - 29564 : _PyMem_DumpTraceback ( ) suggests enabling tracemalloc ( GH - 10510 )
Moshe Zadka < mzadka @ geocities . com > :
merge from 3 . 2
Initial revision
Fix incorrect typo in comment .
Issue : Merge with 3 . 5
bpo - 26441 : Remove references to RFC3339 and ISO8601 from the logging documentation . ( GH - 7297 )
add security
: if a test works in TestMaildir . test_create_tmp
build and - fno - strict overview .
unicode_subtype_new ( ) copies also the ascii flag
Issue : test_curses now raises TclError in the pickle module .
Fix typo in comment .
Added example for the poplib module , contributed by Drew Csillag .
Bugfix by Sjoerd .
Merged revisions 68306 - 68308 , 68340 , 68368 , 68422 via svnmerge from
Merged revisions 72060 via svnmerge from
bpo - 36367 : Free buffer if realloc fails in tokenize . c ( GH - 12442 )
Merged revisions 77389 via svnmerge from
Merged revisions 82769 via svnmerge from
Issue : The ssl module now has support for server - side SNI , thanks to a : meth : ` SSLContext . set_servername_callback ` method .
bpo - 27737 : Allow _Py_hashtable_get_ptr ( ) ( )
Add fclose to newopenfileobject ( ) calls .
revert ssize_t
String method conversion .

