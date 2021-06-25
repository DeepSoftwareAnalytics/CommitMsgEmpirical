: Use subprocess . DEVNULL in webbrowser , instead of opening
Tim Peters : " Audun S . Runde mailto : audun @ mindspring . com wins a
Code style fixup : No need for double ( ( parenthesis ) ) and use { } on an if else .
Merged revisions 86404 via svnmerge from
Fix two small errors in argument lists .
Make regrtest recognize test packages as well as test modules .
Issue : In turtle demos , remove module scope gui and sys calls by
revert unintended change to multiprocessing / queues . py
bpo - 31179 : Make dict . copy ( ) up to 5 . 5 times faster . ( )
Issue : Fixes generation of init import name on Windows .
Issue : Add ns = parameter to utime , futimes , and lutimes .
- fixes telnetlib constants to be one - length byte arrays instead of ints
Fix error messages for buffer objects to say " bytes " instead of " string " .
Issue : Optimize the bytecode for expressions such as ` x in { 1 , 2 , 3 } ` ,
Updated for pending IDLE 0 . 5 release ( still very rough - - just getting
Issue : Fixed incorrect return codes for errors in compile . c .
Wrote section on nested scopes , and moved it to the front
Patch 1353 by Jacob Winther .
Closes : Merged docstring changes from 3 . 3 .
Remove spurious ' u ' .
bpo - 37732 : Fix GCC warning in _PyObject_Malloc ( ) ( GH - 15333 )
Issue 9826 : OrderedDict . __repr__ did not play well with self - referencing dicts .
Get rid of old IDLE . Lib / idlelib rules !
Fixing Issue6557 . urllib . urlopen will quote the space character within urls .
Update for augmented assignment .
Add empty __init__ methods for stateful multibytecodec instances .
merge 3 . 3 ( )
Add John to ACKS
Use shorter version of Connection . isolation_level in sqlite3 . rst
asyncio : Sync with upstream ( compat module )
Issue : Fixed reading out of the buffer in the re module .
: merge with 3 . 1 .
Some ElementTree fixes : import from xml , not xmlcore ; fix case of module name ; mention list ( ) instead of getchildren ( )
Fix bug 9352 where characters were being lost in parsing some short options
bpo - 32940 : IDLE : Simplify StringTranslatePseudoMapping in pyparse ( GH - 5862 )
Include Carbon / Carbon . h in stead of universal headers , if appropriate .
Fix bug - sys . prefix isn ' t always set .
add - - enable - loadable - sqlite - extensions
bpo - 32844 : Fix a subprocess misredirection of a low fd ( GH5689 )
Add tests for the filename .
Final set of changes by Fred before 1 . 4beta3
Add more inline documentation , as contributed in .
It ' s unlikely that future versions will require _POSIX_C_SOURCE
Remove the deprecated statcache module .
Markup fix .
bpo - 38149 : Call sys . audit ( ) only once per call for glob . glob ( ) . ( GH - 18360 )
Flush screen buffer upon console . flush ( ) and output . flush ( ) .
: Fix version tag .
tracemalloc : fix usage of strtol ( ) , value can be LONG_MIN or LONG_MAX on ERANGE
Patch : Implement htmllib . HTMLParser . reset . Fixes .
Merged revisions 55896 via svnmerge from
MacPython - vv equivalent printed incorrect filenames , fixed . Also
merge
Merged revisions 88204 via svnmerge from
bpo - 30860 : Add Include / internal / in " make tags " ( )
Reformatted with 4 - space tab stops .
Issue : Fix the build on PowerPC on Linux with GCC when building with
MERGE : Closes : zipimport . c doesn ' t check return value of fseek ( )
: rewrite docs for locals ( ) and vars ( ) a bit .
Merged revisions 58930 - 58938 via svnmerge from
merge of r58091 from trunk :
: Preserve permissions of mbox , MMDF and Babyl mailbox files on flush ( )
Fix incorrect maintainer name
Followup to c3b47cdea0d1 : document the * write_through * argument to TextIOWrapper .
Typo .
one more place to avoid extra markup ( how many will there be ? )
( Merge 3 . 4 ) Issue : Revert change on selectors , test_selectors failed .
Try to fix buildbot failures with old OpenSSLs .
Qdoffs now exports GWorld new / convert , and Qt uses them .
Only * can be used for varargs argument lists
Fix names import .
Issue : Avoid duplicating socket handles in multiprocessing . connection .
DELETE_FAST should issue an exception when the local variable is undefined .
Issue : Add Misc / NEWS entry .
Fix indentation .
Closes : Merged documentation update from 3 . 3 .
Remove self from the arguments for the function add_type ( ) .
Don ' t use StringType - - always test for basestring .
Add needed # include < ctype . h >
Remove AppVeyor badge now that we don ' t use it in the CI anymore ( GH - 16708 )
- - reset marks if repeat_one tail doesn ' t match
New version by Sjoerd , with support for IRIX 6 audio library .
Add PyEval_CallFunction ( ) and PyEval_CallMethod ( ) .
bpo - 36046 : Fix buildbot failures ( GH - 16091 )
bpo - 33256 : Replace angle brackets around python object repr to display it in html ( GH - 6442 )
Minor beautification ( turn nested - if into a conjunction ) .
Doco update from Sjoerd Mullender .
Merge touch - ups and fixes for ( + port fix to packaging ) and from 3 . 2
Apply patch for issue 3090 : ARCHFLAGS parsing incorrect
Change _tkinter . c compile options .
Add several items
Sorted file names .
Issue : create unit tests for the py_compile module
Issue : Fix description of KEY_PPAGE and KEY_NPAGE constants
Whitespace normalization .
fixed markup errors
Merged revisions 67717 via svnmerge from
merge 3 . 6 ( )
test_time : add tests on HALF_UP rounding mode for _PyTime_ObjectToTime_t ( ) and
Issue : Fix time2isoz ( ) and time2netscape ( ) functions of httplib . cookiejar
Merged revisions 74745 via svnmerge from
: use Host header in asyncore example .
Issue : Blacklist non - text codecs in method API
Make another pass through Misc / NEWS and add stuff .
Pass unquoted string to stat .
Files for second 2 . 1b1 distribution .
bpo - 40286 : Put methods in correct sections . Add security notice to use secrets for session tokens . ( GH - 19870 )
Bugfix candidate .
Issue : merge to default
Closes : avoid shadowing builtin input ( ) in example code .
Work around the conversion of " > > " and " < < " to guillemets . Reported by
add _PyObject_LookupSpecial to handle fetching special method lookup
Typo fixes .
Work around Mac OS X platform encoding issues .
Added pycfbridge . c
Merge
Issue : The zlib module now uses Argument Clinic .
Add a brief section on linking Python as an embedded scripting language .
Update the peephole optimizer to remove more dead code ( jumps after returns )
: Add RFC6532 support to the email library .
Issue : mention historic VS 2008 build dir , too
More additions to whatsnew .
Renamed the repr module to reprlib .
run_suite ( ) : Factor this out of run_unittest ( ) for tests that build
Follow - up to : Fix str . join use , add newlines .
Added more warnings about ni ' s demise .
: idle now doesn ' t hungs if launched as : idle - e < directory >
fix issue23157 - time_hashlib hadn ' t been ported to Python 3 .
Some improvements , e . g . - L / depot / . . . is not needed .
bdecode was already gone in email 5 . This merge adds the test from
Merged revisions 82597 via svnmerge from
add boilerplate so the test modules can be run as scripts
sync whatsnew with 2 . 6
Issue : Fixed crash on deleting ElementTree . Element attributes .
Add importlib . metadata to what ' s new . ( )
Merged revisions 78162 via svnmerge from
use autoconf macro to check for pkg - config ( closes )
Bug : Change Manufacturer of Windows installer to PSF .
Fix compiler warning in mmapmodule . c ( compare signed / unsigned integers )
Closes : use floor division in code example
first cut at unittest version of re tests
SF patch : typo in comment ( unicodeobject . h )
Use unittest . skipUnless to skip the test related to the glibc bug , issue .
Fix markup errors .
Fix SF bug 595838 - - buffer in type_new ( ) should not be static . Moved
Return a handle in stead of a resource .
bpo - 36057 Update docs and tests for ordering in collections . Counter [ no behavior change ] ( )
ANSI - fication
Fix some small markup nits .
Use assertTrue ( ) instead of deprecated assert_ ( ) . ( )
Marc - Andre Lemburg : support pickling Unicode objects , both in text
Merge from 3 . 6 .
: fix a broken link by joining it .
bpo - 37419 : Fix possible segfaults when passing large sequences to os . posix_spawn ( ) ( GH - 14409 )
Added _csv module , and getting rid of macfs .
tarfile . open ( ) with mode ' x ' created files without an end of archive marker .
Fix embarrassing typo and fix constantification of None
replace vars ( ) with locals ( ) and globals ( ) ; 3rd raise arg ; typos
bpo - 33602 : Doc : Remove set and queue references from Data Types ( GH - 7055 )
- Fix doctest results to account for classes being new - style , and thus
Fixed bug that prevented shared libs that are submodules of a
Micro - optimize PyObject_GetAttrString ( )
Group tests about attributes in a separate class .
issue1172 : Documentation of " done " attribute in cgi module .
: fix typo .
Mention the removal of soundex . c
Merge with 3 . 5
Issue : ctypes callback functions no longer contain a cyclic
Fix a couple of glitches identified by Greg Stein .
Patch : Do not defined _XOPEN_SOURCE_EXTENDED on Solaris 10 .
asyncio : Fix BaseEventLoop . _assert_is_current_event_loop ( ) : get_event_loop ( )
This couldn ' t compile on WIndows , due to hardcoded " long long " . Repaired .
Fred ' s sysconfig module .
Neaten - up the itertools recipes .
Merge with 3 . 2
Cosmetic cleanup by Vladimir .
[ Bug ] One of the changes described in the 2 . 3 " What ' s New " document
For MacPython - OS9 verbose is the default .
Add section for ni back in , but comment it out .
Fix test_asyncio / test_events . py : skip IPv6 if IPv6 is disabled on the host
Merged revisions 70960 via svnmerge from
Merge from 3 . 5
Delete a broken threading . local example ( )
restore a generator ' s caller ' s exception state both on yield and ( last ) return
SF , cleanup : remove unused variable
SF bug : test_random fails sometimes
Merged revisions 75472 via svnmerge from
Note the update to Expat 1 . 95 . 7 .
fix a crash with dir ( ) on an uninitialized module
Added tag v3 . 2 . 3rc1 for changeset 7085403daf43
Fixed memory leak on failure .
Patch : Return objects in Tkinter .
bpo - 39272 : Remove dead assignment from _ssl__SSLContext_load_verify_locations_impl ( GH - 17916 )
Toby Dickenson :
bpo - 36961 : Handle positional - only arguments in uparse . c ( GH - 13412 )
Merged revisions 72852 via svnmerge from
If Mark Hammonds win32 tools are not available , try to use the _winreg module
Added tag v3 . 5 . 0rc4 for changeset 2d033fedfa7f
Issue : Speed up Counter operators
Replace mentions of WindowsError
Minor changes .
Merge 3 . 2
Fix typo
replace most calls to emit ' SET_LINENO ' will call to method set_lineno
: Don ' t promote shadowing builtins in the configparser example .
Added Marius Gedminas , for contributions to the datetime code .
merge heads
Fix uninitialized struct member ( GH - 19589 )
bpo - 40431 : Fix syntax typo in turtledemo ( GH - 19777 )
Add byext . py
New built - in module " cl " ( Compression Library ) . Only for Irix 4 . 0 . 5
Issue : Added the " members " parameter to TarFile . list ( ) .
bpo - 39850 : Add support for abstract sockets in multiprocessing ( GH - 18866 )
: remove an unnecessary ' for ' loop from my_fgets ( ) in Parser / myreadline . c .
Additions from Marc - Andre Lemburg < mal @ lemburg . com > , documenting
Close : PyThread_set_key_value ( ) now always set the value . In Python 3 . 3 ,
: fix example .
Add a subsection explaning cipher selection .
- Added DocTestParser . parse ( ) , which parses a docstring into Examples
Added SOLID definition .
Change { \ em . . . } to \ emph { . . . } for consistency with the rest of the Python
Another merge error , seen on bigendian machines
Branch merge
Added a - s option which is useful for narrowing down memory leaks .
Speed up _decimal by 30 - 40 % for numerical workloads by improving the cache
Merged revisions 77576 via svnmerge from
Make test_nntplib more robust
Merged revisions 81621 via svnmerge from
PyObject_Dump ( ) : Use % p format to print the address of the pointer .
Merged upstream changes .
asyncio : sort some methods
Add dependencies for the weakref object .
: Refrain from claiming DOM level 3 conformance in minidom .
[ Bug ] Change docstring of set / frozenset types to specify that the contents are unique . Raymond , please feel free to edit or revert .
Add a crasher for the long - standing issue with closing a file
Issue : Make datetime error tests more lenient .
Darn ! Don ' t divide by zero . Bad fix . : - )
Raise more correct exception on overflow in setting buffer_size attribute of
Merge Py_Cleanup ( ) into Py_Finalize ( ) . Call the various small Fini ( )
A slightly better test of sunaudiodev module . Include the audio data
bpo - 34365 : Update date object documentation ( GH - 8814 )
Preserved one bit in type objects for Stackless .
bpo - 33505 : Optimize asyncio . ensure_future by reordering if conditions ( GH - 6836 )
Merge from 3 . 2
Updated test - runner . bat for new location of Tcl / Tk .
Issue : Extracted shared complicated code in the _io module to new
Add oodles more 8 . 1 Wizard boilerplate . This creates lots of vrbls we
Don ’ t let invalid line in setup . cfg pass silently
Change the list ( ) function to match the documentation in the comment
Initial revision
Issue : Make sure tests don ' t leave the locale changed for any
: use imperative mood in float object docstrings . Patch by Marco Buttu .
Edit the library section , rearranging items to flow better and making lots of edits
Add a format specifier % R to PyUnicode_FromFormat ( ) , which embeds
Raise SystemError when size < 0 is passed into PyString_FromStringAndSize ,
Mac only : two epoch problem workarounds are no longer needed for GUSI
Merged revisions 69324 via svnmerge from
Update for ' ImportWarning '
Forgot the code to initialize the glue for the new CFObj_ routines .
merge
Add regrtest check for caches in packaging . database ( see )
Issue : some additional doc fixes
Add $ PYTHONSTARTUP feature .
detect overflow in combinations ( closes )
Updated the depenencies to reflect changes in the sample code for
bpo - 13927 : time . ctime and time . asctime return string explantion ( GH - 11303 )
Issue : Enhance time . monotonic ( ) documentation
carefully cleanup pointer cache after creating struct pointers
Add missing check of PyDict_SetItem ( ) ' s return value
Merge 3 . 4 ( asyncio )
__rcmp__ no longer gets called on instances . Remove the test for it .
bpo - 31897 : Convert unexpected errors when read bogus binary plists into InvalidFileException . ( )
[ Bug ] Under Windows , socket . read ( ) seems to run into trouble when
Issue : Document that os . spawnle and os . spawnve are not thread - safe under Windows .
Docstring typo fix : Arithmentic - > Arithmetic .
normalize whitespace in Lib / distutils / msvc9compiler . py
Issue : Add Base85 and Ascii85 encoding / decoding to the base64 module .
Issue : Merged fix from 3 . 3 .
Issue : New os . scandir ( ) function , part of the PEP 471 : " os . scandir ( )
* Configure . py : use # ! / usr / local / bin / python
Fix Sphinx warnings .
Fix a bug in the messages for an assert failure where not enough arguments to a string
Fix broken test in test_hotshot . Treating the current directory as an
Issue : Trim trailing whitespace
Localize the function lookup in timeit .
issue9859 : add the missing versionadded tag to the documentation .
Merge : Fix a JSON doc typo
PEP 3101 : Completed string . Formatter class . Reimplemented field_name to object transformation .
Update version .
Got rid of mactoolboxglue . c
Change default ( ) to use getChildNodes ( ) instead of getChildren ( )
A bunch of changes , primarily to command line argument parsing
Added quick hack for bzr
add build machinery for the SSL socket module
Issue : Added Misc / NEWS and versionchanged entries .
Clean up the warnings filter use in test_set .
: import readline in longer example .
Merged revisions 76065 via svnmerge from
Fix test - order - dependend asyncio test failure caused by rev 909ea8cc86bbab92dbb6231668f403b7360f30fa .
Handle calls to win32_ver from non - Windows platform
Re - commit Ping ' s patch to the cgi and cgitb documentation , using the
escape ' \ ' ; thanks to Steve Holden from docs @
Merged revisions 81024 via svnmerge from
Whitespace : break long line
fix typo
Fix indentation level
Merged revisions 67946 via svnmerge from
let the term be linked
Ivan on IRC in # twisted reported this crasher .
bpo - 39424 : Use assertRaisesRegex instead of assertRaisesRegexp . ( GH - 18277 )
Remove a confusing statement in Pickler ' s docstring .
Add initial values to variables to avoid crash in __del__
Fix compiler warning about unused static function .
Merge from 3 . 2
Some pid_t - expecting or producing functions were forgotten in r72852 .
Merged revisions 85429 via svnmerge from
Partial fix for string handling . Null byte TBD .
: remove mention of dbm . bsd which does not exist anymore .
Support OpenUNIX like UnixWare .
: export make_option in __all__ .
Fix declaration - after - statement of d49f65ff4f3c
Remove duplicate ACKS .
Add missing versionadded tag .
Remove the debug print that i accidentally left in .
Bug : Search for shared python library in LIBDIR , not
Issue : improve __classcell__ compatibility
Merged revisions 73717 via svnmerge from
merge
Update pydoc topics .
bpo - 12915 : Improve Unicode support for package names and attributes . ( GH - 18517 )
Issue : Add a Py3k warning when raising an exception that doesn ' t
- Py_DECREF : Add ` do { . . . } while ( 0 ) ' to avoid compiler warnings .
merge
Fixed include paths of PC / VS7 . 1
Remove redundant check for PyBytes in unicode_encode .
Issue : Fixed setting the offset attribute of SyntaxError by
Fixed : shutil . copyfile ( ) was leaking file descriptors when disk fills
Issue : A warning no longer produces AttributeError when the program
Restrict the mode to the lowest four octal positions ; higher positions
: Add missing docs for SMTPUTF8 policy instance .
Complete issue 3083 : add alternate ( # ) formatting to bin , oct , hex in str . format ( ) .
Close : Oops , transposed the digits in the issue number in the previous commit
Patch : Remove FreeBSD 5 specific test , test for ctermid_r , setgroups
dump ( ) : Added asserts that self - > proto is sane .
Step 1 in packaging the toolbox modules and making MacPython more
- Issue : Add a * exists_ok * argument to ` Pathlib . mkdir ( ) ` to mimic
Remove the funky function annotation from numbers . py .
Merged revisions 71976 via svnmerge from
Added regression test for issue24581 .
bpo - 29656 : Handle PR branches in ' make patchcheck ' ( )
Add missing NEWS entry for rev 51803
Add missing svn : eol - style property to text files .
Issue : Failed to build python if configure - - without - threads .
bpo - 29882 : Add an efficient popcount method for integers ( )
bpo - 38631 : Avoid Py_FatalError ( ) in _PyCodecRegistry_Init ( ) ( GH - 18217 )
Merge with 3 . 5 : Issue : tarfile now raises a ReadError if an archive is truncated inside a data segment .
r74963 didn ' t include the merger of _scproxy . c
merge
Typo ( coestring - > codestring ) discovered by Mark Hammond .
Merge from 3 . 5
Issue : Unify ( r ) split , ( l / r ) strip tests into string_tests
Issue 11510 : Fix BUILD_SET optimizer bug .
Fix unittest test discovery for Jython
Issue : Make loop methods reject socket kinds they do not support .
Add 3 . 2 . 1rc2 uuid .
Issue : Fix fd leaks in tests .
Grow that RCS file !
Update doc ACKS .
Nit : round delays to 0 . 1 second .
Changed ' object_filenames ( ) ' to raise exception instead of silently carrying
list comprehensions . see
Feature added by Bill van Melle : when no timezone is present , assume
Issue : Removed unnecessary NULL checks in exceptions . c .
Add news item for
Merge Py Idle changes :
Merge with 3 . 4
Merged revisions 64131 , 64134 - 64141 , 64143 - 64146 via svnmerge from
Update ref docs on slicing .
Lots of cleanup things .
Backport bff16086f03b and bcaaaa00425b .
Try to fix several networking tests . The problem is that if hosts have
Fixed listen ( ) / getsockname ( ) bug
Fix an absurdly invasive test .
Merged revisions 68292 , 68344 , 68361 , 68378 , 68424 , 68426 , 68429 - 68430 , 68450 , 68457 , 68480 - 68481 , 68493 , 68495 , 68499 , 68501 , 68512 , 68514 - 68515 via svnmerge from
bpo - 25409 : Clarify fnmatch and fnmatchcase documentation ( GH - 1535 )
logging : Updated library configuration documentation .
Fixed double hyphens that are rendered to literal en - dashes in the documenation .
Issue : merge
Document the PyMethod_ * type object , functions , and macros .
minsize - - > getminsize .
bpo - 35374 : Avoid trailing space in hhc file name if found on PATH . ( GH - 10849 )
Added socket . getservbyport ( ) , and make its second argument and that of
bpo - 31333 : Fix typo in whatsnew / 3 . 7 . rst ( GH - 5744 )
: merge with 3 . 3 .
Bug : in pyclbr , return full module name not only for classes ,
In open_ftp ( ) , check that retrlen is not None before using it in a % d format !
Fix short file name generation in bdist_msi .
Add : term : s for descriptors .
Update copyright dates in Mac plists .
Remove mentioning of Python 2 . 0 limitations . Fixes .
Improve English phrasing in asyncio task docs .
Minor markup changes .
Fixes issue4653 - Correctly specify the buffer size to FormatMessageW and
SF feature , urlparse fails on imap : / /
Fix Zope URL ( GH - 16880 )
ANSI - fication of the sources .
Fix type in loop stripping ' __ . ' from name
Fixed : now shutil . copytree will let you provide your own copy ( ) function
bpo - 36373 : Deprecate explicit loop parameter in all public asyncio APIs [ queue ] ( GH - 13950 )
AFAICT pydoc / inspect do the right thing again .
Normalize the HTML generated for table headers .
Issue : Document return statement in finally blocks .
Fix issue number typo .
bpo - 39080 : Starred Expression ' s column offset fix when inside a CALL ( GH - 17645 )
Give credit to Adam
Added documentation for the Py_InitModule * ( ) family of functions .
Add a comment explaining the st_symbols cache .
Bump Windows build number for 2 . 3 final .
Add interface to call a Python function ( or other callable ) object
Issue : Raise ValueError on random . triangular call with invalid params .
Issue 10242 . Switching unittest . TestCase . assertItemsEqual to use a collections . Counter under the hood .
symtable_cellvar_offsets ( ) : This leaked references to little integers
Added App module ( appearance manager interface ) .
rapid evolution towards producing real . pyc files ( even though I don ' t
Add docstrings to register , lookup , register_error
Issue : Added test cases for contextlib . ExitStack exception handling behaviour ( Initial patch by Alon Horev )
Merged revisions 81907 via svnmerge from
Merged revisions 80591 - 80592 via svnmerge from
Closes : List Windows XP as an unsupported platform .
Remove duplicate " Reference Guide " in optparse . rst . ( GH - 10372 )
_PyObject_GC_New : Could call PyObject_INIT with a NULL 1st argument .
Add a tiny sleep and additional flush to force the file to really be synced .
Removes lingering references to RAR now that make_zip . py actually makes a ZIP
Add myself to Misc / ACKS
After testing the test on Unix , several improvements :
fix HUGE and MAXVAL ( suggested by Tim Peters )
Document Pyerr_Warn ( ) .
Fix versionadded in unittest . rst .
Merge with 3 . 5
Merged revisions 76593 via svnmerge from
Wrap all test_nntplib methods accessing a remote server in a transient_internet ( )
revert to version 1 . 2
NEWS item for PEP 3115 .
pep8 - fied
Issue : Restored support of bytes paths in os . walk ( ) on Windows .
bpo - 25750 : Fix a compiler warning introduced in GH - 9084 . ( GH - 10234 )
Issue : Fix code __sizeof__ after change .
bpo - 37830 : Fix compilation of break and continue in finally . ( GH - 15320 )
Issue : Use HOST_PYTHON only if it is set .
Properly document multiprocessing ' s logging support , resolve outstanding issues with the custom levels
Issue : Print " fetching < url > . . . " messages only in verbose mode .
Added NULL function pointers for shift and mask ops .
Closes : Merged fix from 3 . 5 .
Convert " srcdir " into an absolute path if that seems prudent . Currrently
Fix availability declaration for popen2 module ; report by Barry Scott
Merged revisions 80337 via svnmerge from
Added Misc / NEWS section for Python 3 . 5 . 0 final .
Merged revisions 70953 via svnmerge from
do not call newmethodobject directly
Issue : Fix test discovery for test_codecmaps_ * .
Merged revisions 80301 via svnmerge from
Issue / Issue : Remove pthread_atfork ( ) handler to remove OpenSSL re - seeding
Merge 3 . 4 ( asyncio )
Add NEWS items for Idle patches and a turtledemo patch .
Revert this change , since it breaks the win32 build :
Change bsddbmodule . o - > bsddbmodule . c .
: fix the : mod : references in email package submodule titles .
Silence DeprecationWarning for cgi . escape ( ) usage in test_cgi .
Issue : Update Windows installer
: merge with 3 . 3 .
: test_sqlite now works with unittest test discovery . Patch by Zachary Ware .
merge 3 . 3 ( )
Don ' t crash if a window no longer exists .
bpo - 42179 : Clarify exception chaining ( GH - 23160 )
Fix incorrect grammar in Misc / NEWS .
NEWS : remove duplicated " Build " section ( GH - 872 )
bpo - 39019 : Implement missing __class_getitem__ for SpooledTemporaryFile ( GH - 17560 )
Issue : Fix unsetenv ( ) test on Windows
Merge .
Issue : Fix patch for 3 . 4 differences .
I believe the intention here was to avoid a global lookup
Docstring consistency with the updated . tex files .
Issue : Update installer Welcome file .
: add base64 . encodebytes and decodebytes as the new spelling of encodestring and decodestring ; deprecate the latter .
Merge 3 . 4 ( typo )
SF patch , fix memory leak of tp_doc
Forward port r62849 .
Additional ExitStack examples , and a few other cleanups for the ExitStack docs
Remove official documentation entry for thread . _count ( ) and make the
ctypes NULL function pointers have a boolean False value now .
Remove over - active replace - string turds .
Issue : Merge with 3 . 4
Merged revisions 83429 , 83436 via svnmerge from
Examples correction in urllib2 howto . Michael Foord
bpo - 37181 : Fix test_regrtest failures on Windows arm64 ( GH - 13872 )
Simplify code in multiprocessing . Connection . send_bytes ( ) .
merge 3 . 3 ( )
fix issue2858 : potential memory corruption when any of the bsddb . db . *
bpo - 36179 : Fix ref leaks in _hashopenssl ( GH - 12158 )
: merge with 3 . 2 .
Fix missing : ref : for idle in doc build .
Issue : retval may be used uninitialized value
Add item
document www interfaces
Issue : Rerun Argument Clinic on all modules
add column offset to all syntax errors
Issue : Actually test relevant assumption
Merged revisions 69459 via svnmerge from
[ SF 710733 - Martin v . Loewis ] Improving source encoding dialog
Remove most references to __members__ and __methods__ , leaving only one pair
document IOBase . __del__ ' s behavior ( closes )
Patch by Tim Peters to speed up colorizing of big multiline strings .
Markup adjustments .
Missing DECREFs when exception is raised in sys . excepthook .
update URL to reflect new website
bpo - 40334 : refactor and cleanup for the PEG generators ( GH - 19775 )
bpo - 35110 : Fix yet few spaces before dashes . ( GH - 10255 )
split_whitespace ( ) : Make sure delimiter is stripped from the beginning
What ' s new in Python 3 . 3 : Rephrase PEP 393 doc
Removed remaining support for Tk versions below 4 . 0 .
Document the optional mode argument ( Z_FLUSH , Z_SYNC_FLUSH , etc . )
Merged revisions 75275 via svnmerge from
closes bpo - 35204 : Disable thread and memory sanitizers for address_in_range ( ) . ( GH - 10442 )
Fix small errors in description of Mac , Windows installation schemes .
Make uuid test suite pass on this box by requesting output with LC_ALL = C .
bpo - 30745 : Fix compiler warnings introduced in bpo - 30730 . ( )
Use unicode
Merged revisions 68750 , 68811 , 68945 , 69157 via svnmerge from
Fix expected error message in PyTextIOWrapperTest
Issue : Backported tests for Scrollbar . activate ( ) and Scrollbar . set ( )
Issue : base64 . b32decode ( ) now raises a binascii . Error if there are
Add example for PEP 557 . ( GH - 5383 )
Remove the new trivial pty . spawn ( ) test as it hangs on many buildbots .
Issue : Writing to ZipFile and reading multiple ZipExtFiles is
Merge with 3 . 4 Issue : idlelib , remove more unused names and imports .
doc update from HEAD . don ' t list " Unix , Windows " as a platform since that
removed resizing functionality .
: Be flexible with appending * . * in shutil . rmtree test case
Fixed None reference leaks in TreeBuilder class ( reported by Neal
Minor clarity edit to contextlib per Guido ' s request .
Issue : Limit length of incoming request in http . server to 65536 bytes
Issue : Prevent crash when passing invalid arguments into msvcrt module .
Issue : Add __qualname__ to functools . WRAPPER_ASSIGNMENTS .
Issue : Merge test_pyexpat from 3 . 4 into 3 . 5
Add pop ( ) to UserDict .
Merge with 3 . 3 : Issue : Link " select " with ws2_32 , the winsock2 lib .
changeit ( ) : Added support for \ deprecated { } .
merge 2 . 7 . 4 release branch ( )
bpo - 30812 : Fix test_warnings , restore _showwarnmsg ( )
Issue : Backport OS X installer updates from 3 . 3 .
Closes : make table of XML vulnerabilities clearer by using " everyday " booleans and explaining the table beforehand .
: merge with 3 . 3 .
Fix some markup errors that prevented formatting , and one that didn ' t .
Small elaboration and typo fixes .
Make new gcc - Wall happy
Issue2297 : Fix a stack overflow in Windows caused by - v and - vv . When python is invoked with - v or - vv under Windows , the process of importing the codec for sys . stderr causes a message to be written to stderr , which in turn causes the codec to be recursively imported . Sometimes the stack overflow exception is swallowed , other times it is not . The bug depends on the particular locale settings of the Windows machine .
bpo - 37027 : Return a proxy socket object from transp . get_extra_info ( ' socket ' ) ( GH - 13530 )
Initial revision
Fixed a typo .
* Add unittests for iterators that report their length
Merged revisions 74873 via svnmerge from
- Issue : Fix library order returned by python - config .
Modify _PyBytes_DecodeEscapeRecode ( ) to use _PyBytesAPI
Issue : Fix an assertion error ( in debug build ) and a crash ( in
Patch : Ignore 2xx response before 150 response .
Issue : SSL contexts will now automatically select an elliptic curve for ECDH key exchange on OpenSSL 1 . 0 . 2 and later , and otherwise default to " prime256v1 " .
Issue : The test_find_mac test in test_uuid is now skipped if the
Issue : Add a get_map ( ) method to selectors .
check for unicode . __mod__
: fix handling of invalid numeric charrefs in HTMLParser .
asyncio : document add_signal_handler / remove_signal_handler , add an example for
Added a note about compiler warnings .
backport test_fileio
Issue : add unittest . main ( ) epilogs to unittest . mock ' s own test modules
* Removed checked_error flag which no longer provides any benefit .
Merged revisions 69419 - 69420 via svnmerge from
Merged revisions 76443 via svnmerge from
merge 3 . 4
PyObject_ClearWeakRefs ( ) is now a real function instead of a function pointer ;
- mark the findCaller 2 - element tuple / 3 - element tuple change as new in 2 . 4
add extra tests to verify that co_varnames is being set up properly
Added SEEK_ * constants . Fixes .
Merge 3 . 2 : test_httplib removes temporary files
Issue : Touch directories only once when extracting a tarfile .
Issue : Remove references to __file__ when part of a frozen application .
made the SGI INET_ADDRSTRLEN define apply for any compiler on Irix . Both the
Update NEWS for a future 2 . 5 . 3 release
Rewritten to use bdb . Bdb as base class .
items ( ) : New method , provided by Gustavo Niemeyer in SF bug .
Issue : Add some notes on behaviour of non - blocking SSL sockets .
Update test_ssl . py to reflect the new approach for writing network - oriented tests in such a way that contention for ports is eliminated , thus allowing multiple instances of the same test to be run in parallel without issue ( import for buildbots to run smoothly ) . This change is in line with the changes made to network - oriented tests in trunk in r62234 . test_ssl . py is an odd case because the py3k version was newer than trunk . When the most recent svnmerge was run from trunk to py3k in r62242 , the older trunk version was brought over on top of the newer py3k version . So , the plan was / is : revert the recent svnmerged version ( done ) , manually make the changes to improve running in parallel ( done with this commit ) , copy this version back to trunk and manually 3to2 it ( to do next ) , check that in , then svnmerge block it . Then when future changes need to be made to test_ssl . py , we can make them against trunk , and hopefully the merge to py3k should be a lot less painful .
Issue : remove undependable and possibly useless test .
Fix broken : ref : in asyncio docs ( GH - 11805 )
: codecs . open ( ) should not convert end of lines on reading and writing .
merge 3 . 3 ( closes )
Added File and Folder , removed macfs .
Skip Montanaro submits a simple patch that makes encode ( ) and decode ( )
code_repr ( ) : use % U to format the filename
Remove last occurrance of PyArg_GetInt . It is deprecated ,
Bug : document that the old slice protocol is still used by builtin types .
: ismount now recognizes mount points user can ' t access .
Add tests for fnmatch . filter and translate .
docs : Document ASYNC / AWAIT tokens ( issue )
Use the thread lock ' s context manager instead of a try / finally statement .
Always call __initialize ( ) , also on PPC
Do the right magic to make \ ABC { } get sorted correctly in the index .
Fixed so - - no - compile is a negative alias for - - compile .
Merged revisions 81231 via svnmerge from
bpo - 17909 : Document that json . load can accept a binary IO ( GH - 7366 )
More exhaustive combinatoric checks .
Add a check for the OpenSSL version number to conditionally compile
Issue : document issue with OpenSSL ' s CPRNG state and fork
Merged revisions 66369 via svnmerge from
Merge 3 . 2
: merge with 3 . 2
Speculative checkin ( requires approval of Gerhard Haering )
: merge with 3 . 3 .
Issue : make the encodefuncs symbol static
Issue : Remove unconditional import of ctypes .
Version 1 . 3 of the Python Unicode Integration proposal .
Issue : Support ' os . mknod ( ) ' in Solaris
err_input ( ) : don ' t encode / decode the unicode message
Changed names , added bridge functions to macfs . fsref objects and
Removed ( empty ) description of \ optional from the " Inline Markup "
asyncio : sync with Tulip
No need to include windows . h twice
SF Bug 667812 : Some Linux distros have Alt and Meta reversed .
Merged revisions 73064 via svnmerge from
Issue : Reintroduce unit tests for the old PyTime API since it ' s still
Issue : The result of open ( ) in Tools / freeze / bkfile . py is now better
asyncio doc : sort methods
Issue : Added shortcuts for utf8 and latin1 encodings .
Bug : Really close underlying socket in _socketobject . close .
Merged revisions 74328 via svnmerge from
Issue 24581 : Revert c9782a9ac031 pending a stronger test for mutation during iteration .
: IDLE no longer raises exception when quickly closing a file .
Clarify that the interruptable popen fixes aren ' t used under Win9x .
Remove outdated and confusing advice about setting maxsize ( GH - 19889 )
merge from 3 . 2
Correctly dispatch Faults in loads ( patch )
add test for inequality
bpo - 40387 : Improve queue join ( ) example . ( GH - 19724 )
Added index entry .
remove hotshot profiler from Py3k
Issue : Divide doctest " Option Flags and Directives " section into two .
In list of sequence comparison examples , " = " should be " = = " . Reported
sched . py : fix test_queue by making sure that queue property returns an orderd list of upcoming events
make sure split ( ) is called before the encoding
test_formatdate ( ) : Remove the unnecessary ldate calculation .
Fix exception catching .
Recorded merge of revisions 83987 via svnmerge from
Patch from Tim Hatch : Remove reference to winreg being the fabled high - level registry interface .
added myself as a maintainer of configparser
Finished bug . Fixed the retrying part to make it
Added explanation of the use of the first program argument passed to the
Build PythonLauncher for MacPython - OSX 2 . 2 as well .
Issue : remove unreachable code in urlopen ( ) .
Typo fix
Merge with 3 . 5
Patch : replace checkline ( ) function parsing with new breakpoint logic :
Fix for issue r1737832
Add some mixed operations between sets and dict views .
SF bug , Spurious errors taking bool of dead pro
bpo - 40874 : Update to libmpdec - 2 . 5 . 0 ( GH - 20652 )
Issue : Properly handle non - integer , non - string arg to SystemExit
Issue : Apply an appropriate fix for importlib based imports
Add parentheses around & & within | | as gcc - Wall advises .
merge heads
Adjusted apply ( ) docstring based on comments from Gerrit Holl
Issue : Increase Treewidge line spacing so lines do not overlap .
Fix for one - letter type in What ' s New In Python 3 . 5 .
add another test case
Link the core with CoreServices , not with Carbon , and don ' t use any Carbon
Issue : Skip required when compiled - - without - threads .
Issue : Enhance os . * chown ( ) testing . Based on patch by Larry Hastings .
Loosen an exception test in test_import to not require an explicit message but instead just the proper exception .
Add a hurriedly - written section on the datetime module
Issue : The py . exe launcher now defaults to Python 3 .
Fixed unichr ( ) - > chr ( )
Added __doc__ attribute ( alias func_doc ) , initialized from first
Issue : cleanup tracemalloc_realloc ( )
Change the look of \ seemodule output to fit in better .
Change isbasestring function as discussed on the cvs list a while ago
Merged revisions 81098 via svnmerge from
: document return value of recv_into ( ) .
Backport r65745 : Issue : Fixed reference leak when occured
Remove debug output from example .
Bump the magic number to avoid sharing bytecode between 2 . 3 and 2 . 4 .
Issue 4790 : Eliminate unnecessary work from heapq ' s nlargest ( ) and nsmallest ( )
Updated super ( ) as per http : / / www . artima . com / weblogs / viewpost . jsp ? thread = 208549
listobject . c : added optional cmp function to list . sort ( ) .
Implemented PEP 405 ( Python virtual environments ) .
Add destrpy ( ) method .
Issue : fix test_time for Mac OS X and OpenIndiana
: merge with 3 . 3 .
Preliminary documentation for the curses module by Moshe Zadka , with
: Some more Windows related tuning to shutil . rmtree tests
Issue : don ' t raise OverflowError for complex ( ' 1e500 ' ) . Backport of r72803 .
Mention list . sort ( )
Oleg Broytmann ' s support for RFC 2231 encoded parameters , SF patch
bpo - 31966 : Fixed WindowsConsoleIO . write ( ) for writing empty data . ( GH - 5754 )
- wrap a long line
Issue : logging : Fixed possible race condition in lock creation .
Add signatures to the docstring of functions added to imp by PEP 3147
Marc - Andre Lemburg < mal @ lemburg . com > :
Issue : Uninstalling pip does not leave behind the pyc of
Merged revisions 71722 , 71725 via svnmerge from
correct signature for tzinfo . dst ( ) in examples ; thanks to Daniil Shved from docs @
Correct docs for empty raise when no exception is active .
Use start_new_session in the webbrowser module rather than a python os . setsid preeexec_fn .
Issue : Merge string_tests cleanup from 3 . 5
- Example of HTTP PUT Request using httplib
Merged revisions 77821 via svnmerge from
Issue : Fix BaseEventLoop . stop ( ) docstring , incomplete sentence .
check for error conditions in _json
Changed the # error to a # warning when checking gcc versions , and
_parsebody ( ) : Fix for the new message / rfc822 tree structure ( the
Merged revisions 65826 via svnmerge from
Remove gratuitous unicode character .
Fix core dump from pow ( x , y , 0 ) .
Issue : Merge from 3 . 5
Update example of non - blocking SSL code for the new finer - grained exceptions
bpo - 38467 : Fix argument name of typing functions ( GH - 16753 )
Move constructors , add some casts to make C + + compiler happy . Still a problem
Issue 26911 : fix import ( other problems remain ) .
Merge with 3 . 2
- New Extension : CodeContext . Provides block structuring hints for code
Issue : Added tests for CLI of the zipfile module .
An extensive test suite for the email package .
Fix typo ( missing $ in )
Issue : merge from 3 . 4
The ' test ' target ought to depend on ' all ' , not just on python $ ( EXE )
Simplify markup .
Add / GS - flags for AMD64 .
Add NEWS entry I forgot to commit .
Only __dict__ and __class__ are read - only instance attributes
Give test_multiprocessing better chance of avoiding timeout failures on Windows
Merge 3 . 4
SF patch by Walter Dörwald : auto - guess content - type header for
: specify default maxsize value ; patch by Floris Bruynooghe
Fix memory error in test_ucn .
Patch : Don ' t include empty path component from registry ,
bpo - 32030 : Add _PyCoreConfig . module_search_paths ( )
Fix syntax warnings in tests introduced in bpo - 35942 . ( GH - 11934 )
__new__ is a static method
Merge : : clarify index entry for source file encoding declaration .
Recreated after source changes .
bpo - 38034 : Fix typo in logging . handlers . rst ( GH - 15708 )
Issue 9222 Fix filetypes for open dialog
Fix for issue 7149 : a regression in 2 . 6 . 3 that causes an exception when
Fixed : distutils . sdist . add_defaults now add files listed in package_data and data_files
Marc - Andre Lemburg < mal @ lemburg . com > :
Initial revision
Remove the use of ' inspect ' from ' warnings ' for detected deprecated use of the
Merge 3 . 4 ( asyncio doc )
: clarify that webbrowser is not meant for file names .
Adjustments to the text of the UnixMailbox description . Added
[ 1197218 ] test_locale fix on modern linux
Merged revisions 66887 , 66891 , 66902 - 66903 , 66905 - 66906 , 66911 - 66913 , 66922 , 66927 - 66928 , 66936 , 66939 - 66940 , 66962 , 66964 , 66973 via svnmerge from
Major upgrade by Vladimir Marangozov .
( Merge 3 . 4 ) Issue : Fix compilation on Windows
Merge 3 . 2
markup fix
SF patch : Tests and minor bugfix for uu module
Patch : Improve PanedWindow .
Merged revisions 65459 , 65472 , 65481 , 65518 , 65536 , 65581 , 65609 , 65637 , 65641 , 65644 - 65645 via svnmerge from
merge with 3 . 4
Recorded merge of revisions 74650 via svnmerge from
Add some more basic tests to validate the argument checking of
Issue : Fix a transient test_timeout failure .
Document and test the resolution of issue 3445 ( tolerate missing attributes in functools . update_wrapper , previously implemented as a side effect of the __annotations__ copying patch ) and implement issue 9567 ( add a __wrapped__ attribute when using update_wrapper )
merge
Merge : : Remove misleading error message to fix regression .
Changes in anticipation of stricter str vs . bytes enforcement .
Tagging 2 . 6a1 . . . again !
Issue : Replace ' master ' with ' parent ' in tkinter . messagebox calls .
Clarify a number of issues about the file - like object API based on
closes bpo - 34474 : Python / bltinmodule . c : Add missing NULL check in builtin_sum_impl ( ) ( GH - 8872 )
merge with 3 . 4
Improve accuracy of sequence and mapping checks .
Issue839496 : SimpleHTTPServer should open all files in binary mode .
Merged revisions 77499 via svnmerge from
bring is_integer into tested existence
bpo - 1635741 : Port multiprocessing ext to multiphase init ( GH - 21378 )
Remove ' created by ' lines ; people can use CVS for this , and the information is often out of date
settscdump is definitely an implementation detail
Merge with 3 . 3
Issue 1776581 . Minor corrections to smtplib , and two small tests .
Solaris ' / dev / null is a symlink . The device test now uses stat instead of lstat to compensate
* import . c ( MAGIC ) : Changed magic word to avoid confusion about exec
updating to cElementTree 1 . 0 . 5 ( step 2 of 3 )
I give up : unless I write my own strftime by hand , datetime just can ' t
Issue : Install a copy of template C module file so that
Drop xmllib in favor of SAX .
Merge with 3 . 4
Slightly more lenient pdbtrack prompt recognition , for people who use
: ur ' \ U0010FFFF ' used to raise in narrow unicode builds .
Merged revisions 70356 via svnmerge from
Merge with 3 . 2
test_multiprocessing removes temporary files
whitespace
Fix xover implementation according to Fred Lundh .
Speed - up search for hidden names by using a set instead of a tuple .
add hotshotmain . py ref
: document sqlite version strings , use correct one in test .
Modifed for new included expat .
Add a test for the final branch in repr . Repr . repr1 ( ) , which deals with
Add some more files from 1031 that I missed .
Add new encodings
flesh out open ( ) docstring
Issue : Avoid compiler warnings when compiling readline with libedit .
More changes needed to make things work once bytes and str are truly divorced .
jcompile ( ) : inherit the CO_GENERATOR_ALLOWED flag from the ' base '
merge
Merge spelling fixes from 3 . 4 into 3 . 5
Change the way hex type - ins are displayed . The old way was way too
: Popen . communicate ( ) argument must be bytes .
Merged trivial doc fix from 3 . 2 .
made interrupted static again
Merged revisions 68776 - 68777 via svnmerge from
PEP 314 implementation ( client side ) :
Add _subprocess . c
Issue : Use the first coding cookie on a line , not the last one .
Trimmed trailing whitespace .
Merged revisions 77060 via svnmerge from
Fix for issue 5558
Issue : ftp : / / gatekeeper . research . compaq . com / and ftp : / / ftp . debian . org /
Issue : Removes hard dependency on hg . exe from PC / build . bat
Merged revisions 80789 via svnmerge from
bpo - 27319 , bpo - 31508 : Document deprecation in Treeview . selection ( ) . ( )
Issue Suppress compiler warnings during clean . Patch by Zachary Ware .
bpo - 39114 : Fix tracing of except handlers with name binding ( GH - 17769 )
Document that some of LZMAFile . __init__ ( ) ' s args are keyword - only .
Clarified the error cases and Unicode handling of PyString_AsString ( ) ,
SF patch : Fold tuples of constants into a single constant
get skipIf from the right place
: fix a bug in pstats . add_callers ( ) and add a unit test file for pstats .
Issue : Skip test_with_pip when threading module is not available
Issue : Fix opening empty archives and files .
use PyMac_Yield ( ) instead of PyMac_Idle ( )
os . fsync ( ) should be used to ensure that data is written to disk
Merged revisions 85503 via svnmerge from
Fix a few urllib bugs ( NameErrors ) .
bpo - 29596 : Improve clinic howto documentation ( GH - 1710 )
Merge 3 . 6 ( issue )
Merged minor documentation update from 3 . 3 .
whrandom should come immediately after random , not before .
Generalize to macintosh ; forget Think 3 . 0 .
Add a What ' s New entry for bpo - 35459 . ( GH - 12706 )
Upgrade pyexpat to expat 2 . 0 . 0 ( http : / / python . org / sf / 1462338 ) .
Merged revisions 82779 , 82855 , 83740 , 83789 - 83791 , 83797 - 83801 , 83803 , 83811 , 83827 , 83844 via svnmerge from
Added release dates ( in some cases detemined by looking at the mtime
bpo - 33987 : Add master ttk Frame to IDLE search dialogs ( GH - 22942 )
Brought some module variables into synch with pickle . py ' s current values .
Repair portability of sign extension when reading signed ints on boxes
merge 3 . 2 heads
Add a deprecation warning to reflect the documented deprecation of the
Add _PyUnicode_HAS_UTF8_MEMORY ( ) macro
Finish a sentence that was left half - written !
Note about enhancements to unicode ( ) .
os . getcwd ( ) is returning str8 ; sys . path items are str .
Closes issue : Enhancements / fixes to pure - python datetime module
Fix typo .
Rewrite PyBytes_FromFormatV ( ) using _PyBytesWriter API
Issue : Fixed reading out of the buffer in the re module .
Added Andrej Krpic . ( Thanks for issue patch . )
- fixed default value handling in group / groupdict
Patch : Fix bugs in the plural handling .
bpo - 8677 : use PY_SSIZE_T_CLEAN in socketmodule . c ( GH - 12467 )
Add a couple of files that are necessary based on getting PCbuil8 to work .
typo : privide - - > provide
Issue : Workaround for missing symbol " gethostbyaddr_r " on Android .
Make naming consistent
Use an unsigned cast to avoid a warning in VC + + .
Issue : rework the performance question in the programming FAQ
Attempt to fix Solaris buildbot failure on test_locale
Merge from 3 . 6
bpo - 35516 : platform . system_alias ( ) don ' t replace Darwin ( GH - 11207 )
Issue : Fix test_cmd_line and test_faulthandler for my previous change
Fix leaking a RuntimeError objects when creating sub - interpreters
test_socket : use context managers to close directly the socket
Un - installing of AE handlers fixed ( AE is very picky that you specify
Fix comments and whitespace .
Issue : Fix not getting the locale ' s charset upon initializing the interpreter ,
bpo - 34973 : Fix crash in bytes constructor . ( GH - 9841 )
_PyUnicodeWriter_WriteStr ( ) now checks that the input string is consistent
merge 3 . 3 ( )
Remove duplicate recipe .
Fix minor markup typo ( )
unlink ( ) would normally be found in the " os " module , so use it from there .
- clean up table markup for readability
bpo - 40077 : Convert _bz2 module to use PyType_FromSpec ( GH - 20960 )
: fix duplicate name and remove duplicate test . Patch by Vajrasky Kok .
Issue : Fix recursive usage of cursors . Instead of crashing , raise a ProgrammingError now .
Closes : codepoint - > code point
( partparse . py ) : Small fix to do_datadesc ( ) : in some odd cases the name of
SF bug [ ] Looks like a unary minus bug .
bpo - 29661 : fix contradiction in timeit . Timer . autorange ' s docstring ( GH - 331 )
Rename classes to StackViewer ( the widget ) and StackBrowser ( the toplevel ) .
Patch : Always extract member names from the tarinfo .
small changes by Soren Larsen
Directly read working copy data to obtain list of properties .
More updates to whatsnew .
Issue : Add a __sizeof__ method to struct . Struct .
Make TextIOWrapper ' s documentation clearer by copying the newline argument ' s description from open ( ) .
Stop raising OverflowError on underflows reported by libm ( errno = = ERANGE and
Add Guilherme Polo .
Merged revisions 84472 via svnmerge from
Merged revisions 60210 - 60233 via svnmerge from
Issue : Re - implement imp . load_module ( ) in imp . py .
Generate the clear ( ) method manually ( calling __GLclear ( ) , another
Turn section references into proper cross - references .
Add example to the documentation for calling unittest . mock . patch with create = True ( GH - 11056 )
merge with 3 . 3
Close : Add get_wch ( ) method to curses . window class
Issue : typo fix . This commit is accompanied by an apology for all Liechtensteiners out there , and a thanks to Eric Araujo for noticing .
Refactor the common code out of the posix and windows listdir
Added some prototypes .
Add a function _Py_ReadyTypes ( ) which initializes various and sundry
* Grammar : corrected old typo ( class instead of ' class ' )
Issue : The pure Python pickle implementation now treats object lengths as unsigned 32 - bit integers , like the C implementation does .
Add traceback . format_exc ( ) .
Got rid of ifdefs for long - obsolete GUSI versions and other stuff that is now standard ( appearance , interned strings )
Issue : Fix embedded readline ( ) hang on SSL socket EOF .
Merged revisions 76485 via svnmerge from
Updated version from Donn Cave < donn @ oz . net > .
merge 3 . 3 ( )
Merge 3 . 6
Fix typo in whatsnew
Backport PEP 3110 ' s new ' except ' syntax to 2 . 6 .
Armin Rigo ' s fix & test for
Fixed : added ARFLAGS for the archiver command .
Open the source file in universal newline mode .
Merged revisions 71560 via svnmerge from
Merged revisions 81640 via svnmerge from
Merge from 3 . 3 .
merge 3 . 5 ( )
Add check for the PyArg_ParseTuple format , and declare
Close : register memoryview with Sequence ABC
Issue : Catch PermissionError in tests when spawning a non existent program
Issue : add tests for inet_ntoa , inet_ntop , inet_aton and inet_pton .
bpo - 32030 : Add _PyMainInterpreterConfig . executable ( )
Minor grammar re - wording
Added hints about building your own DLL .
purported linux portability patch ( Oliver Andrich )
_is_string_secure ( ) : This method is no longer used ; removed it . ( It
re - checkin with " ISO - 8859 translation " turned on .
flush stdout before writing to stderr in print_error ( )
Add HAVE_OLD_CPP to list of known defines
Manual py3k backport : [ svn r74316 ] Issue : Fix io . BytesIO to not accept arbitrary keywords
Issue : cProfile now displays the module name of C extension functions , in addition to their own name .
Supply long - missing docs for random . seed ( ) . Extensive rewrite of module
replace reference to debugger - hooks
Remove some old comments adequately covered in the code .
bpo - 38529 : Fix asyncio stream warning ( GH - 17474 )
Add support for package data .
Modernize clean and funny targets .
Merge 3 . 4 ( )
Issue : Fix default values of sched . scheduler . enter arguments were modifiable .
Issue 2246 : itertools grouper object did not participate in GC ( should be backported ) .
Cosmetic fix to project files .
This should fix issue2632 . A long description of the two competing
bpo - 36402 : Fix threading . _shutdown ( ) race condition ( GH - 13948 )
Merged revisions 58862 - 58885 via svnmerge from
SF bug : codecs . StreamRecoder . next ( ) now reencodes the data it reads
compile with - std = c99 instead of - std = gnu99 ; use kiddie - gloves with bluetooth / bluetooh . h ( )
Issue 2235 : __hash__ is once again inherited by default , but inheritance can be blocked explicitly so that collections . Hashable remains meaningful
Publicize the GC untracking optimization
Merge tag ' v3 . 8 . 0a4 '
docs . capi : Fix tp_as_async doc
Add four library items
Merge heads .
general cleaning up
Include all licenses of the packages that we include .
Add AsyncContextManager to typing module documentation . ( GH - 6822 )
Issue : map_next ( ) uses fast call
Fix ' refleak ' introduced by fnmatch cache purge tests .
Use PyDict_Copy ( ) and PyDict_Update ( ) instead of using PyObject_CallMethod ( )
Add deprecation warnings for modules as documented
Issue : Define HAVE_LONG_LONG as 1 .
Issue : Document that zlib and gzip accept a compression level of 0 to mean ' no compression ' .
Reword message
Issue : datetime . timedelta constructor now rounds microseconds to nearest
Issue : Do not include accu . h from Python . h .
Merged revisions 77477 - 77478 , 77481 - 77483 , 77490 - 77493 via svnmerge from
Add a few imports to examples .
reapplied r74493 ( after fix has been backported )
correct ontosys url - closes 1167329
Issue : When using " python3 - m test - - testdir = TESTDIR " , regrtest doesn ' t
bpo - 34580 : Update sqlite3 examples to call close ( ) explicitly ( GH - 9079 )
This is the last time I fix binhex . If it breaks again it goes in the dustbin .
Plug a leak with calling something other than a function or method is
# Checkpointing X / Emacs 20 ' fication of this mode .
Document list . sort ( ) changes
correct case in python16 . dsp to allow build with VS6
Issue : merge from 3 . 3
Issue : Make ' filter ' argument in tarfile . Tarfile . add ( ) into a
importlib . abc . SourceLoader . get_source ( ) was re - raising SyntaxError and
keep the buffer object around while we ' re using it ( closes )
Added new exception classes :
Issue : Change _decimal to accept any coefficient tuple when
Revert debugprint code in r77607 .
Merged revisions 77999 via svnmerge from
When a ctypes C callback function is called , zero out the result
Another change suggested by Donn Cave - - forget " step 5 " , which
: Update the documentation of os . fdopen ( )
Reap children before the test starts so hopefully SocketServer
Issue : Fix Tkinter Resources link
No need to emit co_lnotab item when both offsets are zeros .
oops , rename pymonotonic_new ( ) to pymonotonic ( )
test_debug in test_tkinter / test_text no longer fails when wantobjects is false .
Merged revisions 78426 via svnmerge from
Issue : os . stat_float_times ( ) has been deprecated .
SF Bug # 941346 , AIX shared library fix
added CoreGraphics to build list
Add \ exception markup
Issue : Fix PyErr_SetImportError ( ) , handle PyDict_SetItemString ( ) failure
Merged revisions 66179 , 66283 via svnmerge from
This is release candidate 2
Issue : Improve examples in tkinter documentation
Merged revisions 77136 via svnmerge from
don ' t use subprocess . call with PIPEs as the child can fill the pipe buf and
Revert " bpo - 35565 : Add detail to assertion failure message in wsgiref " ( GH - 11317 )
Issue : Allow detect_tkinter to look for Tcl / Tk 8 . 6 .
Issue : Unknown escapes in regular expressions that consist of ` ` ' \ ' ` `
Hacked to support the notion of " negative alias " options , to handle
Christopher Fandrich < cfandrich @ 8cs . com > :
Added new function k_lopsided_mul ( ) , which is much more efficient than
clear error state properly
Issue : Document that ssl . RAND_egd ( ) is not available with LibreSSL
fix indentation ; looks like all I managed to do the first time is make things uglier
: merge with 3 . 2 .
Issue : test_bz2 now works on non - Windows platforms without bunzip2
Bump Windows resource version macro for 2 . 3 final .
Issue : Change " context manager protocol " to " context management protocol " .
Fixed restore_files ( ) ; added reset_files ( ) ; made these more flexible .
Remove buildno .
Added note ( ) message to Page class - - this was used but didn ' t exist .
Fix typo .
Make _ctypes_test . c compile on Windows .
The identify ( ) description didn ' t match the code . Fixed .
Fix typos and improve grammar in threading . Barrier docstrings ( GH - 12210 )
Change the approach to pickling to use __reduce__ everywhere . Most
Doc sidebar : 3 . 6 has moved to security - fix mode . ( GH - 11810 )
( Merge 3 . 2 ) Issue : python - gdb . py is now able to retrieve more frames in
- update some comments
Close : Fix select . select ( ) on OpenBSD 64 - bit
Merged revisions 77267 via svnmerge from
let ' s not return NULL from functions that should return ints
Issue : Add PATCH to wsgiref . validator . Patch from Luca Sbardella .
merge
Fix Python 2 . 6 support in python - gdb . py
Issue : Try to fix refcount bug . Empty and 1 - byte buffers are always shared .
OK , reop is safe again .
Issue : Add odictobject . h on Windows .
PyErr_Format ( ) does not return a new reference ; it always returns NULL .
Fix whitespace .
Issue : Allow building Python without external libraries on Windows
Generate new style exception initializer ( Jack ) .
Whitespace normalization .
compile doesn ' t accept code objects
changes for complex and power ( * * ) operator
If we use frameworks we should include CoreServices / CoreServices . h , not the
fix condition ( )
Issue : Refleak in PyMethodObject repr .
Close : fix a race condition in siginterrupt ( ) tests
: consistency update , and backport avoiding ` ` None > = 0 ` ` check from py3k .
( libtime . tex ) : Merged in changes from Tamito Kajiyama .
Merged revisions 85299 via svnmerge from
Issue : Use specific asserts in sqlite3 tests .
the name of the slot of nb_multiply not nb_mul
detect alpn by feature flag not openssl version ( closes )
Issue : Merge 3 . 5 .
Issue : Fixed contributors name .
bpo - 33305 : Improve SyntaxError for invalid numerical literals . ( GH - 6517 )
Issue : Merge 3 . 4 into 3 . 5 ; adjust new tp_as_async field
Fix a small nit in the error message if bool ( ) falls back on __len__ and it returns the wrong type : it would tell the user that __nonzero__ should return bool or int .
Add entry for ColorPicker documentation as well .
At Jeff Rush ' request , add Py_BEGIN / END_ALLOW_THREADS around call to
Handle error
Urmpf . Quality control on this patch lapsed a bit . : - (
Fix a crash : when sq_item failed the code continued blindly and used the
Minor usage fix .
bpo - 15999 : Clean up of handling boolean arguments . ( GH - 15610 )
Issue : Improve the documentation of the dict constructor .
Get rid of debug print statement .
merge heads
When performing a POST request , i . e . when the second argument to
Issue 24298 : Fix signature ( ) to properly unwrap wrappers around bound methods
Decreased memory requirements of new tests added in issue21448 .
Issue : Cleanup internal_connect ( ) in socketmodule . c
bpo - 40204 : Fix reference to terms in the doc ( GH - 21865 )
[ Patch by Scott Dial ] urllib and urllib2 were using
Factored out the code that creates argument lists and formats for PyArg_Parse
Added test_socketserver and test_unicode_file to tests expected to be
Fix markup
Add py3k warnings for object , type , cell and dict comparisons . This should resolve issue2342 and partly resolve issue2373 .
Fix typos and clear up one very odd bit of wording as pointed out by
Merged revisions 73864 via svnmerge from
Merged revisions 79184 - 79185 via svnmerge from
Remove issue 19081 Misc / NEWS entry since it isn ' t entirely fixed yet .
Issue , reported by Nguyen Quan Son , fix by Fredruk Lundh :
bpo - 39056 : Fix handling invalid warning category in the - W option . ( GH - 17618 )
Issue : Fix - - with - tsc build failures on x86 - 64 that resulted
Issue : Move function type annotations into docstrings .
Issue : change the incorrect " deprecation " of ftplib dir ( ) and nlst ( )
datetimetz_astimezone ( ) : Speed optimizations - - although I ' d rather
Issue : Fix O ( n * * 2 ) behaviour when adding or removing many subclasses of a given type .
Fix overzealous patterns in . hgignore , which used to hide files containing e . g . " python " .
Issue : Delete Python 2 . 2 mention from distutils documentation .
Closes : Merge with 3 . 5
: merge with 3 . 3 .
Issue : One more check for set_pre_input_hook ( )
A rewrite for better Python - ish style
merge 3 . 3
Moved a bunch of routines from " blacklisted " to " graylisted " , as they _are_
Whitespace cleanup
Issue : signal . sigtimedwait ( ) timeout is now a float instead of a tuple
Merged revisions 82204 via svnmerge from
Close : Fix typo in asyncio doc . Patch written by Brett Cannon .
Issue : Avoid using an obsolete instance of the copyreg module when the interpreter is shutdown and then started again .
: Fix str . capitalize ( ) to correctly uppercase / lowercase titlecased and cased non - letter characters .
New test module for complex math module
Issue 16398 : Use memcpy ( ) in deque . rotate ( ) .
merge
Move Windows stuff to 2 . 2 , so CVS builds won ' t interfere with 2 . 1
asyncio / tests : Fix deprecation warning
encoding returns bytes objects
Merged revisions 81115 via svnmerge from
Issue : in int - to - decimal - string conversion , reduce intermediate storage requirements and relax restriction on converting large integers . Patch by Serhiy Storchaka .
bpo - 33950 : Remove IDLE htest spec for a deleted file . ( GH - 7881 )
Issue : Resurrect inspect . getargspec in 3 . 6 . Backout a565aad5d6e1 .
Issue : Preserve subprocess fork exception when preexec_fn used
Note that __getitem__ ( ) may receive a slice object as the index ;
Rewrite asyncio test to be more meaningful ( )
New posix functions open , close , dup , dup2 , lseek , read , write , fstat , pipe .
bpo - 28393 : Update encoding lookup docs wrt bpo - 27938 ( )
Add notes from python - dev about readonly dictionaries .
Issue : bytesobject . c should not have its own private defines for stringlib macros . Also removed unused defines and include for localutil . h .
fix run - on sentence
: clarify that copyfile ( ) doesn ' t take a target directory .
bpo - 26701 : Improve documentation for the rounding special methods . ( )
Minor change to test new CVS reporting
New solution to the " Someone stuck a colon in that filename ! " problem :
Fix duplicate doc entry for SSLContext . get_ca_certs ( )
Fullbuild now maintains BUILDNO for mac builds
Test all three EXT opcodes , and move these tests into
Issue : Fix a file descriptor leak when importing a module with a bad encoding .
Merged revisions 63828 via svnmerge from
bpo - 36921 : Deprecate @ coroutine for sake of async def ( GH - 13346 )
Enhance the performance of two important Unicode character
Simplify example
Added a GetControlRect ( ) method to controls which returns the bounding rectangle . To my surprise this call is missing from the C API . . .
Fix to remove relative imports from multiprocessing .
Added docstrings ( contributed by Martin von Loewis ) .
Remove test_listener_client to remove possible race condition in the multiprocessing tests
Issue : Make traceback reporting ignore any errors when printing out
Quite a few fixes to make the library and test suite more robust when
Whoops ! We just discovered that Gordon ' s revamp of this module was
Issue : Remove deprecated float coercion from struct module , along
bpo - 33766 : Document that end of file or string is a newline ( GH - 7383 )
SF bug : ThreadingMixIn / TCPServer forgets close
os : fsencode ( ) , fsdecode ( ) and os . environ ( b ) internal encode - decode methods
Slightly reorganized so it is useable from another program , and
Fix formatting
Really make bztar support in shutil conditional .
The set types can also be called without arguments .
Issue : Improve exception message of warnings . warn ( ) for bad category .
Merged Misc / AtheOS - NOTES into the platform - specific section . Rewrote the
Add more tests for pdf ( ) and cdf ( ) ( GH - 12190 )
Issue : Use path_error ( ) in more posix functions , especially in Windows
upgdaded the test program
Bug : operator . isSequenceType ( ) now returns False for subclasses of dict .
Hint about [ \ ] trick to avoid quad backslashes .
Undo some ( but not all ) of the more lenient acceptance of
bpo - 1875 : Raise SyntaxError in invalid blocks that will be optimised away ( GH - 13332 )
Edit Menus section of Idle doc . Make menu entries match current Idle in order
bpo - 30919 : shared memory allocation performance regression in multiprocessing ( )
Changes by Trent Mick ( of ActiveState ) and Dale Nagata to support
Merged revisions 78979 via svnmerge from
Add versionadded tags to new math functions .
glob ( ' anything * / ' ) would fail because isdir is in os . path , not os .
Make sure the icons / directory is created even if only one document is
bpo - 31933 : fix blake2 multi - byte params on big endian platforms ( )
Merged revisions 68920 via svnmerge from
Include Windows . h for BOOL . 2 . 7 and 3 . 2 already had this due to
Add new syntax for specifying filenames on break and clear commands .
Issue : The tokenize module , IDLE , 2to3 , and the findnocoding . py script
Update comment and make accurate .
Second part of fix for bug [ ] Python 2 . 2b2 bdist_wininst
" proces " - - > " process " ; reported by Gerry Wiener < gerry @ ucar . edu > .
merge
Remove one use of UserDict . UserDict
Alas , the thread support for Tk didn ' t work . Withdraw it , until I
Merge from 3 . 3
Add test cases for internal SHA3 helpers
merge from 3 . 5
Add notes to the asctime ( ) and ctime ( ) descriptions to make it
Hyphenate " built - in " for consistency .
Patch , bug , make mmap . mmap ( - 1 , length ) work the same
Wording nits .
Closes : add unittest . main ( ) to test_email / test_inversions .
Merged revisions 77542 via svnmerge from
Use stderr instead of stdout for warning .
: roll back on error during add so mailbox isn ' t left corrupted .
merge 3 . 4
Trimmed trailing whitespace .
Adding Wing 4 ( IDE ) project file
Kill div , mod and divmod on complex ( already deprecated in 2 . x ) .
move unprefixed error into . c file
Merged revisions 75342 via svnmerge from
Merged revisions 76360 via svnmerge from
Issue : Fixed a gcc warning with - Wstrict - overflow .
Added Jeff Rush ' comments
# Issue 11424 : merged fix from 2 . 6 .
merge from 3 . 4 - clean up the subprocess docs warning - soup and
Wording changes
rearrange declarations somewhat
bpo - 41044 : Generate valid PEG python parsers for opt + seq rules ( GH - 20995 )
bpo - 34521 : Fix tests in test_multiprocessing_spawn to use correctly CMSG_SPACE ( GH - 9613 )
Added 38 , 642 missing characters to the Unicode database ( first - last
fix ws
Fix a bunch of problems from merges .
Fix warning : ‘ ptr2 ’ may be used uninitialized in this function
Merged revisions 76217 via svnmerge from
bpo - 37124 : Fix reference leak in test_msilib ( GH - 13750 )
Fix out of bound read in UTF - 32 decoder on " narrow Unicode " builds .
Issue : When called without an explicit traceback argument ,
Fix issue number in comment .
Merge with 3 . 5
Poke and hope for Tcl version : now settable through _TCLMINOR_
_PyImport_LoadDynamicModule ( ) encodes the module name explicitly to ASCII
Use boolean name
Issue : Cleanup _ssl . c
bpo - 35642 : Remove asynciomodule . c from pythoncore . vcxproj ( GH - 11410 )
removing one Mac hack and add another :
Dead code removal from _hashopenssl . ( GH - 11379 )
Fixes issue : subprocess . poll ( ) now properly handles errno . ECHILD
Deprecated torgb for removal in 3 . 0 .
Define NDEBUG for releae builds , just like Python .
Issue : remove some details that only apply to the 3 . x version of this module and cross reference the relocated warning about the dangers of invoking the shell with untrusted input
bpo - 29972 : Skip tests known to fail on AIX ( )
Fix markup in Counter note .
Merge with 3 . 5
Fix test_pydoc failure introduced by 2f09a6980e1a ( issue ) .
Fix typo .
Merge fix for from 3 . 3
Merged revisions 70499 via svnmerge from
More nits .
Update more instances of has_key ( ) .
Started updating information about defining attributes on types .
bpo - 35713 : Split _Py_InitializeCore into subfunctions ( GH - 11650 )
Added decl for Py_InteractiveFlag .
bpo - 35766 : Change format for feature_version to ( major , minor ) ( GH - 13992 )
Remove references to make targets deleted in eef7899ea7ab
bpo - 31819 : Add AbstractEventLoop . sock_recv_into ( ) ( )
Get rid of listextend_internal ( ) and explain why the special case
Documentation for the py_compile module .
Patch : Suppress COPYRIGHT if site . py is not read . Fixes .
TarFile . __init__ ( ) no longer fails if no name argument is passed and
Add an item
Text moved to PEP 339 .
: Fix memory leak after key function failure in sort
Fixes issue : use a list to accumulate the value instead of repeatedly concatenating strings .
Merge with 3 . 4
Using CleanImport to revert a reload of the os module doesn ' t work due to function registrations in copy_reg . The perils of reloading modules even for tests . . .
Ditch some dead code in test_unicode_file .
bpo - 37540 : vectorcall : keyword names must be strings ( GH - 14682 )
Issue : SSLv3 is now disabled throughout the standard library .
More cmd . exe exploitation .
Clarify the + and b mode characters for open ( ) a bit .
Merged revisions 74713 via svnmerge from
` self ` doesn ' t exist here
# Added doc string .
Properly guard against special - casing .
Make test_parser pass after merge .
Issue : skip test_dynamic_key when run in non - interactive mode .
Added getattr and setattr built - in functions .
Issue : Fix " setup . py register - - list - classifiers " .
Added gete ? [ gu ] id functions
Added PySys_SetPath .
Patch : quit and exit can now be called from the interactive
Added frexp , ldexp , modf , fmod .
Fix SF buf 404774 submitted by Gregory H . Ball
Issue : Fix documentation for PyIter_Next ( ) .
Trigger a few error cases in Modules / _codecsmodule . c .
Issue : test_pep277 formats filenames with ascii ( ) on error
[ Bug ] Fix typo in example
locale . bindtextdomain ( ) : use PyUnicode_FSConverter ( ) to parse the filename
Changed way * . ind are forced to exist ; the files are now reset to contain
Fix wording ( 2 . 6 . 1 backport candidate )
Issue : replace dead link with version - specific doc link .
Issue : BZ2File and bz2 . decompress ( ) now support multi - stream files .
Issue : If the system readline library is linked against
- Issue : Fix __pycache__ pyc file name clobber when pyc_compile is
Per the discussion in issue6882 , backport the try / finally work that was done to the py3k version ( mostly in r59477 , I think ) .
preparing for 2 . 5c2
add some recent releases to the list
tutorial typo fix
Merge for issue
Add bytes methods documentation .
Fix typo - - - e takes a flag , - h doesn ' t . Somehow the colon and the h
Merged revisions 74196 via svnmerge from
PyObject_Dump ( ) encodes unicode objects to utf8 with backslashreplace ( instead
Patch : Fix pdb ' s " ignore " and " condition " commands so they trap the IndexError caused by passing in an invalid breakpoint number .
Patch : Disable more Unicode API if Unicode is not used .
By default , ! = returns the opposite of = = , unless the latter returns
Merge 3 . 4 ( create_stdio )
Issue 24004 : Support Awaitables ( pep 492 ) in @ asyncio . coroutine decorator
Add a useful docstring to enumerate .
e . g . - > e . g . ,
Correct the name of a field in the WIN32_FIND_DATAA and WIN32_FIND_DATAW structures .
Added extern declarations for reference count admin debug functions .
Typo : destuction - - > destruction
Bug : Fixed semaphore errors on AIX 5 . 2
Remove buffer ( ) usage in the socket module by just slicing directly on the
Issue : Fixed possible leaks in the ctypes module .
( Merge 3 . 1 ) Issue : When calling a method descriptor or a slot wrapper
do_cmd_arabic ( ) : Override the l2h - provided version to remove that
- Issue : Fix the incorrect shared library extension on linux . Introduce
: merge with 3 . 3 .
Version bump for Python 3 . 4 . 6 .
: rephrase minidom documentation to use the term " minimal " instead of " lightweight " . Patch by Éric Araujo .
Fix , non - keyword argument following keyword .
Accept only the system default encoding when converting Python
Document transfercmd ( ) , ntransfercmd ( ) , delete ( ) , rmd ( ) , size ( ) .
Issue : xmlrpc now raises ResponseError on unsupported type tags
Closes : Improved test_logging . Thanks to Diana Joan Clarke for the patch .
Remove extra space in docstring
test_ftplib : skip check_hostname test when SNI is not available
Merged revisions 77595 via svnmerge from
added a few more __all__ lists
Spell emdash with three hyphens .
Added SetFolder method which sets Standard File initial folder
Fix context manager use in posixpath . join ( ) tests
Fix . Update some sections of Windows FAQ .
bpo - 36362 : Avoid unused variables when HAVE_DYNAMIC_LOADING is not defined ( GH - 12430 )
make this subversion artifact empty
Hoist the global dummy lookup out of the inner loop for set_merge ( ) .
Catch only ImportError
[ Patch from David M . Cooke ]
Add more links to the ' What Next ? ' section ; rewrite to be briefer
bpo - 33786 : Fix asynchronous generators to handle GeneratorExit in athrow ( ) ( GH - 7467 )
( Merge 3 . 4 ) Issue : make the zlib module " ssize_t clean " for parsing
closes bpo - 39922 : Remove unused args from four functions . ( GH - 18893 )
documentation wording fix for issue4903
Issue 7117 : Replace PyOS_ascii_strtod with PyOS_string_to_double in stropmodule as part of short float repr .
suppress compiler warnings in _ctypes_test ( )
bpo - 38341 : Add SMTPNotSupportedError in the exports of smtplib ( )
bpo - 38870 : Implement round tripping support for typed AST in ast . unparse ( GH - 17797 )
Drop unused import in unittest package .
Revise cheeseshop example so that the order of the keyword output is
Added missing walk ( ) function
Issue : Document binascii . crc_hqx ( ) implements CRC - CCITT
test_logging coverage improvements .
Re : move array bounds check before array access .
The value is the dotted module name to the command class .
Don ' t catch interrupts in getpass ( ) - - the finally clause will reset
Added glue routine for PyMac_BuildFSSpec , PyMac_GetFSRef and PyMac_BuildFSRef .
Merged revisions 65622 via svnmerge from
Issue : Prevent missing ' getentropy ' declaration warning on macOS .
bpo - 33608 : Factor out a private , per - interpreter _Py_AddPendingCall ( ) . ( gh - 13714 )
Python 3 . 9 . 0a1
Issue : Fixed possible integer overflow in bytes and bytearray
Issue : pickle now uses an identifier to only create the Unicode string
NEWS entry for .
Merged revisions 82879 via svnmerge from
Fix whitespace .
Added more documentation about the executable argument .
bpo - 40462 : fix variable and function names ( GH - 19832 )
remove special metadata
Issue 4293 : Make Py_AddPendingCall ( ) thread safe
Issue : Fix http . server to again handle scripts in CGI subdirectories ,
call_trace ( ) : Add an additional parameter - - pointer to a PyObject *
Use - x64 flag .
Since bdist_wininst . py contains the installer executable , it had to be
: make generated HTML more valid
Issue : array . fromfile ( ) failed to insert values when EOFError was raised .
Clarify the note for UserList
Fiddle comments and variable names in whichmodule ( ) .
This commit was manufactured by cvs2svn to create tag ' r14beta3 ' .
Issue : turn the DeprecationWarning from float arguments passed
Online help facility for the Python interprter and other interactive
Fix a bogus end = ' ' here .
Whitespace normalization ( tabs - > spaces )
Issue : TarFile expose copyfileobj bufsize to improve throughput
Added Py_PROTO macros for SunOS 4 . x / bin / cc .
* Objects / classobject . c , Include / classobject . h : added __getattr__
Catch DistutilsFileError in addition to DistutilsExecError in ' setup ( ) ' .
changes refs to SO_ * symbols
Add markup for True / False . Will backport
bpo - 20104 : Fix leaks and errors in new os . posix_spawn ( GH - 5418 )
fix ACKS : alphabetic order and UTF - 8
Issue : The Py_IS * macros from pyctype . h should generally only be
: no more < > around addresses in VRFY or EXPN
Reflect file renamings for the Python Library Reference .
Merge from 3 . 4
Test utility to look for bad stacksize calculations .
- update to reflect that base classes can be class types as well as
Drop unused variable .
Merged revisions 87238 via svnmerge from
Added libthreading . tex ; this is mostly the contents of threading_api . py ,
For sets with cyclical reprs , emit ' . . . ' instead of recursing .
Typo in the classic mlte build line .
make bytearrayobject . o depend on the stringlib headers
Use raw strings for docstrings with backslashes in ASCII diagrams
Minor wording change in section title .
merge 3 . 3
Issue : documentation fixes related to the cmp removal .
Merged revisions 84489 via svnmerge from
Add NEWS item for new string methods .
Issue : Comment in middle of shell command fails on BSD and OS X
remove support for byte literals ; a new feature
test_on_error ( ) : Rewrite so it works on WinXP too . Unsure about 95 / 98 / ME .
And another .
Remove unused imports .
Merged revisions 75911 via svnmerge from
e macresource . need ( ) to open the W resource file in the application init
Merged revisions 78620 via svnmerge from
Fix various potential buffer overrun problems .
Add a test for extension modules in the old - style record file
Merged revisions 72792 via svnmerge from
change a few uses of the threading APIs
Bump version to 3 . 4 . 0 alpha 0 .
Stop test_tcl ' s testLoadTk from leaking the Tk commands ' loadtk ' registers .
Merged revisions 72361 via svnmerge from
Merged revisions 87374 via svnmerge from
Patch : Pass install - purelib to idleinstall .
Use assertRaisesRegex instead of deprecated assertRaisesRegexp .
Merge heads
Add warning / error handlin for problematic nested scopes cases as
Forward - port r52358 :
bpo - 24914 : mention Python supports multiple paradigms in the FAQ ( )
Merge directory chooser into tkFileDialog .
Thomas Wouters : Test script for openpty ( )
Issue : Fixed buffer overrun in binascii . b2a_qp ( ) and binascii . a2b_qp ( ) .
bpo - 30911 : Fix a crash in json on platforms with unsigned char ( )
Skip the test requiring ctypes if ctypes is unavailable .
Miscellaneous minor fixes of Misc / NEWS formatting . ( )
merge from 3 . 2 - Fix closes Issue12576 - fix urlopen behavior on sites which do not send ( or obsfuscates ) Connection : Close header .
Issue : Added tests for xml . sax . parse ( ) and xml . sax . parseString ( ) .
ensure that availability information is on its own line at the end of the function docs
Issue : Merge UnsupportedOperation fixes from 3 . 5
( py - no - outdent - re ) : Added return , break , raise , continue since we
Issue 3068 : Move idlelib . configDialog action button creation into a separate
Patch : Enable extensions on NetBSD 2 . 0 .
Changes pyvenv . cfg trick into an actual sys . path file .
Patch : test case for bug in difflib .
bump api version
Merged revisions 75209 via svnmerge from
Merged revisions 75192 via svnmerge from
: Fixed user paths in sysconfig for posix and os2 schemes
Minor : improve one test name , address pyflakes warnings
changed \ x to consume exactly two hex digits , also for unicode
- Various tweaks to shut up compiler warnings .
Convert socket methods to use METH_O and METH_NOARGS where possible .
Cleanup : the in operator already returns a bool , no need to bool ( ) it again
Issue : Fix the unit test
Fix makefile changes for python - config .
Merged revisions 74802 via svnmerge from
Add temporary code to fix the automatic doc build failure .
Issue : add NEWS entry .
towards 1 . 6b1
bpo - 35621 : Support running subprocesses in asyncio when loop is executed in non - main thread ( GH - 14344 )
Addresses issue 2802 : ' n ' formatting for integers .
Issue : BaseHTTPRequestHandler again logs response code as numeric ,
SF bug and : Clarify use of __all__ .
Closes : put a link to the CHM download page on the docs download page .
Issue : Clarify that pool methods can only be used by parent process .
Rationalized MS ifdefs
Recorded rollback of revisions 84857 , 84870 , 84875 via svnmerge from
Backout change e8f44ebacda7052267318cecf5b6f128d35add17 . Reverting the test
Merged revisions 76776 via svnmerge from
bpo - 38809 : Windows build scripts use python . exe from virtual envs ( GH - 17164 )
Replace outdated optimization with clearer code that compiles better .
Expand documentation about type aliases and NewType in the typing module .
Try to strengthen test_httpservers ( issue )
I ' ve been waiting 8 years for KSR to re - emerge from bankruptcy . If they
Rationalize MS # ifdefs
Issue : The webbrowser module now supports Firefox 36 + and derived
- Issue : Reject coroutines in asyncio add_signal_handler ( ) .
: Remove remnants of debugging code
Use f - strings in asyncio - task code examples ( GH - 10035 )
Remove the outdated notion that multithreading doesn ' t work well on Unices .
bpo - 24048 : Save the live exception during import . c ' s remove_module ( ) ( GH - 13005 )
Put current date into POT - Creation - Date ; leave PO - Revision - Date for
bpo - 33519 : clarify that . copy ( ) is not part of the MutableSequence ABC ( GH - 6965 )
Merge with 3 . 4
Merged revisions 81724 via svnmerge from
Merge 3 . 4 ( asyncio test )
Issue : fix the documentation mis applied patch .
bpo - 13826 : Clarify Popen constructor example ( GH - 18438 )
Issue : fix a bug in the peepholer that led to non - portable pyc files between narrow and wide builds while optimizing BINARY_SUBSCR on non - BMP chars ( e . g . " \ U00012345 " [ 0 ] ) .
Issue : Add support for arbitrary - X options on the command - line .
Removes unused files from PC folder .
Issue : Merge getpagesize doc from 3 . 4 into 3 . 5
Issue : Add missing imports and update tests .
Cast the arguments to PyString_AsStringAndSize ( ) to silence compiler warnings
SF patch 670012 : Compatibility changes for _strptime . py .
Added tests for sample modules .
Fix indentation ( reindent . py ) .
Merge from 3 . 6 .
Fixed another missed ConfigParser import rename .
bpo - 41056 : Fix reference to deallocated stack in pathconfig ( Coverity ) ( GH - 21013 )
Add test coverage reporting for C files ( GH - 7773 )
Delete references to _SC_AIO_LIST_MAX ; it appears that that symbol was
Fix appendChild ( ) and insertBefore ( ) ( and replaceChild ( ) indirectly ) when
Added regression test for SF tracker bug : AttributeError in
Fixed viewrect / destrect calculation
Clarify that named tuples do not have to subclass tuple .
Issue : Fixed build with Estonian locale ( python - config and distclean
Oops , undo previous change , which wasn ' t supposed to escape from my
More factorization .
[ Bug ] Document correct argument range for pair_content ( )
Issue : Improve test coverage of datetime . tzinfo
bpo - 39999 : Improve compatibility of the ast module . ( GH - 19056 )
A getopt . c for your amusement
bpo - 37369 : Fixes crash when reporting fatal error ( GH - 14468 )
Minor stylistic and consistency cleanup .
transform izip_longest
More has_key ( ) fixes .
Issue : Silence Visual Studio warnings .
further work on config system and config saving
Merged revisions 86006 via svnmerge from
make the output of _test ( ) suitable for use in the regression test
Do not release unallocated Tcl objects . Closes and .
Added new . isalpha ( ) and . isalnum ( ) methods which provide interfaces
Issue 964437 Make IDLE help window non - modal .
fix a bug in the previous commit . don ' t leak empty list on error return and
use assert [ Not ] IsInstance where appropriate
Issue : Re - enabled test_bug_1333982 in test_dis , which had been
: clarify timeit documentation to say that setup statement isn ' t timed
Issue : fix code samples in extending / newtypes . html for PEP - 7 compliance
Issue : Fixed segfault in curses ' s addch ( ) with ncurses6 .
modernized ColorPicker somewhat
Add what looks like a necessary call to PyErr_NoMemory ( ) when PyMem_MALLOC ( )
Fix some bugs in c - api / arg . rst documentation
Always add space after RFC ; reword paragraph
Implement suggestion from Lawrence Kesteloot in PR , to change the
merge with 3 . 4
Fix issue : Support unpickling of exceptions pickled by Python 2 .
Fixed : Distutils uses the tarfile module instead of the tar command now
typo
Another demo from Thomas Gellekum ( seasonally - themed , too )
bpo - 37008 : make mock_open handle able to honor next ( ) ( GH - 13492 )
Improve getting the mailbox from the environment .
Added the - - dist - dir option that the " bdist_ * " will use to control where
Issue : Fix crash in test_zlib on bigmem buildbot .
Update comment about surrogates .
Compile site - packages with - t , not - tt .
Issue 17826 . Setting an iterable side_effect on a mock created by create_autospec now works
Add Josiah .
Use the cwd when the empty string is found in sys . path . This leads to
Note that True and False are pickable objects
audiodev . py : Mac port .
Issue : Added save / restore for argv and os . environ to runtest_inner
bpo - 31234 : Join threads in test_hashlib ( )
New batch from Fred
Fixed error in new comment .
SF patch : The UTF - 8 and the UTF - 16 stateful decoders now support
Issue : Fix callbacks race in asyncio . SelectorLoop . sock_connect .
Optimization : move RFC defined network constant construction out of
Issue : Remove unused assignment and redundant GDB CLI arguments
Some more notes about bytes / string filename APIs .
Issue : Remove intobject . h from MSVC project files .
merge from 3 . 6
Merged revisions 76771 via svnmerge from
allow byte literals
Merged revisions 85459 - 85465 via svnmerge from
Merge Py Idle changes
bpo - 33649 : Edit asyncio eventloop doc - second pass ( GH - 9233 )
New file
Nick Mathewson : test suite for the class browser support module .
Fixup handling of free variables in methods when the class scope also
Issue : Removed ' indefinitely ' from the proleptic calendar description .
merge
Merged revisions 85990 via svnmerge from
Issue : Add example of customizing User - Agent via add_header ( )
Add more general warning against the SGI optimizer .
Issue : Fix docstrings for binascii . ( un ) hexlify
Add declarations for PySys_ResetWarnOptions ( ) and
Use title case
Fix curses module for strict SysV implementation ( without has_key function )
Avoid codec spelling issues by just using the utf - 8 default .
Add more news about iterators .
- changed hash calculation for unicode strings . the new
Added concrete example of when separate build and install might be
use native tenary condition
Merged revisions 83533 via svnmerge from
more markup fix .
Issue : Fixed integer overflow issue in " namereplace " error handler .
Issue : updated documentation on exception ( ) method and function .
Unicode database updated ; use SVN instead of CVS
* * * empty log message * * *
Documented PYTHONOPTIMIZE ; by Marc Lemburg .
Issue 10593 : Adopt Nick ' s suggestion for an lru_cache with maxsize = None .
Issue : save / restore exception on filename encoding
Merge : fix typo , improve sentence flow
Issue : The type of long range iterator is now registered as Iterator .
Avoid the Py_UNICODE type in codecs . c
Docs : Mention that Py_UNREACHABLE was added in 3 . 7 ( )
Issue : run autoreconf to update configure .
bpo - 36342 : Fix test_multiprocessing in test_venv ( GH - 12513 )
Merged revisions 85689 via svnmerge from
Remove sys . exc_type , sys . exc_value , sys . exc_traceback
xml . dom : fix typo , drop unused imports .
Merged revisions 77733 via svnmerge from
Refactor astgen . py with 2to3 .
Issue : OS X now failing on the second setrlimit ( ) call
Issue : limit year to [ 1 ; 9999 ] for strftime ( ) on Solaris
Add a few tests to test_count ( ) to increase coverage in
bump version to 2 . 7a3
Close : Document that Python 3 sets LC_CTYPE at startup to the user ' s preferred locale encoding
remove imputil demo
Merged revisions 59843 - 59863 via svnmerge from
Add an assert .
Use MacOS . OutputSeen ( ) on menuselections . Removed KeepConsole code again . This solution is better : it treats a menu selection in the same way as input to stdin ( i . e . as an ack of the output that went before ) .
Add a reverse ( ) method to collections . deque ( ) .
check the errno in bad fd cases
Merged revisions 86732 via svnmerge from
[ doc ] Teach 0 - args form of super in Programming FAQ ( GH - 22176 )
- Wformat is needed by gcc 4 . 8 ( closes )
include both the GIF and PNG icons in the HTML package
- Better way to find site - packages
bpo - 36167 : fix an incorrect capitalization ( GH - 14482 )
Added Makefile . pre . in , my generic makefile
Fix closes Issue12359 - Removing a confusing sentence from the previous change .
Merge .
Defer compilation of regular expressions until first use .
bpo - 23835 : [ docs ] configparser converts defaults to strings ( )
revert
Issue : Improved logging documentation based on suggestions in the issue .
Patch : SYNC_FLUSH the zlib buffer for GZipFile . flush .
Hmm , this test has failed at least twice recently on the OpenBSD and
Updates to the OS X installer ( merge from 3 . 5 )
Issue : Fix module exception list and __file__ handling in freeze .
add aifc
Add copyright . tex and boilerplate . tex to the common dependencies for . dvi
Merge from 3 . 2
Doc for PEP 341 , needs improvement
Added typechecking to the individual python - > CF converters , so we can use them in the CF object initializers safely .
FreeBSD platform specific modules
Oops , default type and creator were reversed .
Issue : Popen . communicate ( ) on Unix now accepts strings when
give rational filenames to some of the generated HTML
All the exception building related stuff has been moved out of this
( Merge 3 . 3 ) regrtest . py : Fix another typo in the usage of the faulthandler module
Issue : Old - style classes ( not just instances ) now support weak
Patch : Fix typo .
merge 3 . 2
Make sure subsections are formatted into HTML pages with reasonable names .
( Merge 3 . 3 ) regrtest . py : Fix typo in the usage of the faulthandler module
Initial revision
Merge heads .
Closes : Merged fix from 3 . 6 .
Merged revisions 78198 via svnmerge from
Enhanced HTTPHandler documentation .
Drop bf_getbuffer / bf_releasebuffer from stable ABI ,
Feature request : Add os . path . devnull .
Fix several changelog items to use PEP markup .
Issue : Fix possible crash in cPickle on invalid input . Patch by
take into account keyword arguments when passing too many args
Fix typo in Doc / whatsnew / 3 . 9 . rst ( GH - 17372 )
fix decoding in _stringify to not depend on the default encoding
Add extension management to __all__ .
merge 3 . 1 ( )
In search of TLS 1 . 1 bug : add debugging output in verbose mode
Various items
bpo - 30822 : Deduplicate ZoneInfoTest classes in test_datetime . ( )
: merge with 3 . 2 .
The rest of the changes by Trent Mick and Dale Nagata for warning - free
Removed ' or long integer ' from bin , oct , and hex docstrings .
fix typos in whatsnew
Merged revisions 67246 via svnmerge from
Issue : add reference to IANA timezone database ; thanks to Georg / Nick suggestions
: Treat EROFS like EACCES when making a ' file is read - only ' decision
Issue : Convert IDLE search dialogs to using ttk widgets .
Implemented show_choice ( ) ; moved items around a little bit .
Remove duplicate entry
Backport fix of ( only OpenKey applies here ) .
Merge issue : Add function annotation example to function tutorial .
Revert a premature patch for issue ( changeset 846bd418aee5 ) .
Do the check for no keyword arguments in __init__ so that
Issue : fix regression in test_pathlib when Windows has symlink support available ( i . e . in administrator mode ) .
( python - cc - style ) : Oops , we decided to use TABs for 8 spaces where
Eliminate latin - 1 encoding
Issue : correct example for captured_stdout ( )
Silence warnings from ' make suspicious ' to make the docs buildbot happy
No need to preprocess the header files - use ctags - I flag instead to
bpo - 31948 : Fix broken links in msilib docs ( GH - 4397 )
Made all c_ * symbols static .
Added HAVE_STDARG_PROTOTYPES and signed
Patch : Fix bug , where shlex . shlex hangs when it
Kill reduce ( ) . A coproduction of John Reese , Jacques Frechet , and Alex M .
Internal refactoring of convertsimple ( ) and friends .
Issue whitespace
Merge changes up to 1 . 10 from PyXML :
Less red ink ( warning - > note ) and add link to def of side - by - side assembly .
Remove unused variable .
Merge with 3 . 5
Fixed bug in line - number finding for examples ( DocTestParser wasn ' t
Fix SF , test_resource failure on alpha / 64bit
Deleting Python 1 . 5 project and Wise file .
Stacks based on lists work better and read more clearly when they
Move over to using assertRaises as a context manager for importlib tests .
Make test . test_support . catch_warnings more robust as discussed on python - dev . Also add explicit tests for itto test_warnings .
: only create a new MIME boundary if we don ' t already have one .
Markup nit : use \ envvar to mark environment variables .
Added { Get , Set } PopupData calls to get at the data for popup menu controls .
Issue : some syntax errors from * args or * * kwargs expressions
Issue : dev_urandom ( ) now calls py_getentropy ( )
url2path for NT
Issue : Got rid of nonneeded use of PyUnicode_FromObject ( ) .
Issue : Remove tests . txt of test_asyncio
Update typing . py and test_typing . py from upstream ( https : / / github . com / python / typing ) ( merge 3 . 6 - > 3 . 7 )
Issue : Skip test_import . test_unwritable_directory on FreeBSD when run as
Lots and lots and lots . . . . .
Issue : Prevent doctests from failing when sys . displayhook has
Mention specific exceptions instead of OSError where appropriate .
Issue : Expat parser now correctly works with string input not only when
Issue : try to fix test_startfile ' s inability to clean up after itself in time .
Issue : tracemalloc : report tracemalloc_log_alloc ( ) failure to the caller
bpo - 38500 : Add _PyInterpreterState_SetEvalFrameFunc ( ) ( GH - 17340 )
Remove single " . " components from pathnames , and return os . curdir if
Demote warnings to notices where appropriate , following the goal that as few " red box " warnings
Handle testlist_safe as if it were testlist .
Generate libffi ' s Makefiles again to be able to run the libffi testsuite
Fix bug introduced in rev . 46806 by not having variable declaration at the top of a block .
Add sanity assertions in some import lock code ( issue ) .
Issue : Change assert to TypeError in from_buffer / _copy ( )
Issue , part 6 : fix test_hashlib for missing threading module
: refactor tests to use subtests and proper assert methods . Patch by Vajrasky Kok .
Merge fix from 3 . 3 into default .
bpo - 34926 : Make mimetypes . guess_type accept os . PathLike objects ( GH - 9777 )
Document new " cmdoption " directive .
Issue : Fix test_regrtest when Python is compiled in release mode
- merge heads
Cleanups , and conversion of assert to assertEqual ( )
merge from 3 . 2 - Fix closes issue12581 - Increase the urllib . parse test coverage ( cases applicable to 2 . 7 ) . Patch by Petter Haggholm .
DisposPtr - > DisposePtr
Issue : Load all ciphers and digest algorithms when initializing
Unfold some lines that were previously folded but don ' t need to be
bpo - 38070 : Enhance _PyObject_Dump ( ) ( GH - 16243 )
Merged revisions 83140 - 83141 via svnmerge from
Fix - up syntax problems .
Merged revisions 78318 via svnmerge from
Fix typo
bpo - 37207 : enable vectorcall for type . __call__ ( GH - 14588 )
Revert SF ( rev 53655 ) : dict . update ( ) does * not * call __getitem__ ( ) or
bpo - 29723 : Consistently configure sys . path [ 0 ] ( )
Add note to asyncore / asynchat recommending asyncio for new code .
Merged revisions 76538 , 76559 , 76882 - 76883 , 76886 , 76891 - 76892 , 76920 , 76924 - 76925 , 77081 , 77084 , 77086 , 77092 via svnmerge from
Initial revision
Skip a test of an old API on ARM Linux that isn ' t supported by the OS .
Keep the definition of i consistent between set_lookkey ( ) and set_insert_clean ( ) .
Merged revisions 79740 via svnmerge from
Move deprecated functions at the end of their respective sections .
Check for identical types before comparing objects to see if they
Issue : make test_create_connection ( ) handle ENETUNREACH .
Adds a subprocess . check_call_output ( ) function to return the output from a
Merged revisions 88613 via svnmerge from
Issue : Update rlcompleter test for new 3 . 6 behaviour
convert test_audioop to unittest . Thanks to Giampaolo Rodola .
Issue 16774 : Add a new itertools recipe ( suggested by Alexey Kachayev ) .
Issue : merge fix from 3 . 3
whatsnew : deprecation of HTTPConnection ' s strict parameter .
type
run autoconf ( step forgotten in r72446 )
Get rid of the classes RModuleLoader and RModuleImporter - - these were
fix weird sh - bang line
Merged revisions 83536 , 83546 - 83548 , 83550 , 83554 - 83555 , 83558 , 83563 , 83565 , 83571 , 83574 - 83575 via svnmerge from
Fix copy & paste error : call title instead of count
Patch from Perry Stoll : support for Windows .
Merge heads
Moved the description of the tzparse module to the " Obsolete " section
http : / / bugs . python . org / issue6267
Issue . Added isgenerator ( ) and isgeneratorfunction ( ) to
First Checked In .
fix compiler warnings
Add various items
merge 3 . 3 ( )
document that cadefault does nothing now
Merged revisions 85678 via svnmerge from
Merged revisions 77866 - 77867 via svnmerge from
tag 2 . 7 alpha 4
MAde a few things more orthogonal and did some cleanups :
Corrections and additions to the documentation for POP3 and IMAP library
packaging : Add the project directory to sys . path to support local setup hooks .
_split_ascii ( ) : lstrip the individual lines in the ascii split lines ,
squash compiler warning on Mac OSX 10 . 3
: Drop email __version__ string . It no longer means anything .
Issue : 3 . 3 Branch - Remove unreachable code in urllib . request
As Tim Peters points out , ` ` from string import * ' ' should not set re to None .
Merged revisions 76212 via svnmerge from
merge 3 . 2
Moved prototyes for fmod and pow , and don ' t use them for THINK C .
Merge from 3 . 3 : Fix placement of shell = True warning in subprocess . Popen ( ) docs .
Merged revisions 67900 - 67901 , 67919 , 67928 , 67984 , 67991 - 67993 , 68106 - 68108 , 68110 via svnmerge from
Merge issue : Fix collapsing of code sample in tutorial .
bpo - 25777 : Wording describes a lookup , not a call ( GH - 15573 )
Small imporvements , added mpz and md5 to list of modules ,
Fix for issue 8476
Issue , : test_pathlib uses support . rmtree ( ) instead of
: merge with 3 . 3 .
bpo - 41199 : Docstring convention not followed for dataclasses documentation page ( GH - 21413 )
A Python float is a C double ; redeclare defaulttimeout as such ; stops
Doc changes on the trunk will not be in Python 2 . 2 , so let ' s call it 2 . 2 + .
Fix comment in test_gdb
Remove redundant import
Issue : Move idlelib . EditorWindow . HelpDialog deprecation warning
Added cvsfiles . py and treesync . py
ANSI - fication of the sources .
Patch by Vladimir Marangozov to include the function name when
Got rid of the internal datetimetz type .
Added shlex and netrc modules ; added warning about urllib change
bpo - 41054 : Simplify resource compilation on Windows ( GH - 21004 )
Merge from 3 . 3 .
Issue : Removed private function _PyUnicode_CompareWithId .
Remove mention of the old - X command line switch .
Patch : Add name and value arguments to
Merged revisions 78384 via svnmerge from
Issue : Optimize deque . rotate ( )
test_urlparse : add tests for encoding and errors arguments
correctly handle descrs with __missing__
Replace dead code with an assert .
Revert " bpo - 30487 : automatically create a venv and install Sphinx when running make ( GH - 4346 ) " ( )
Bugfix for issue3885 and ' DB . verify ( ) ' crash .
bpo - 36144 : Document PEP 584 ( GH - 18659 )
Small PEP8 correction .
merge heads
Issue 14814 : Docs work showed some more cases of networks pretending to be addresses and highlighted the weird approach to implementing the ' is_whatever ' properties . Impl now illustrates far more clearly that networks have a property if both their network and broadcast addresses have that property
For the title page , insert the package version number before the date .
bpo - 34228 : Allow PYTHONTRACEMALLOC = 0 ( GH - 8467 )
Set TCL_LIBRARY before import _tkinter . Suggested by Kirill Simonov .
: merge with 3 . 3 .
Spewed a bunch more verbiage .
merge 3 . 4 ( )
Merged revisions 85193 via svnmerge from
completely detabify unicodeobject . c
Merge for porting HOWTO update
: fix example in re doc .
Issue : The UTF - 7 decoder now accepts lone surrogates
Clean up from - import handling .
Michael Hudson : With the ( cool ! ) new call syntax , the longest opcode
Fix most of the bsddb3 tests .
Issue : Add restart support for binary upload in ftplib . The
Disable the test for importing very long lists for MacPython : it triggers
check for str . __mod__
Issue : Path . glob ( ) now raises a ValueError if it ' s called with an
Merge with 3 . 3
bpo - 39573 : Add Py_SET_TYPE ( ) function ( GH - 18394 )
Merged revisions 85008 , 85011 , 85240 via svnmerge from
Merged revisions 78056 via svnmerge from
Fixed unicode ( ) to use the new API PyUnicode_FromEncodedObject ( ) .
bpo - 36919 : make test_source_encoding . test_issue2301 implementation - independent ( )
Fixed IDLE tests after changing re error messages ( issue ) .
Remove unnecessary < p > from top navigation bar .
Issue : The readable ( ) , writable ( ) and seekable ( ) methods of BytesIO
Fix deprecation warning in test_univnewlines . py
Use OpenSSL 1 . 0 . 0a on Windows .
Added readdisplay ( ) routine
Merge 3 . 5
Describe startswith ( ) / endswiith ( ) change ; add reminder about wsgiref
argparse documentation updates ( including updates to optparse and getopt documentation that were promised in the PEP )
Add support for extension modules in 3 . 0 deprection command .
Get this module to compile with bsddb versions prior to 4 . 3
Removed partially broken { verbatim } support .
Added an extra frame in widget heirarchy in case I want to add
On Linux , one sometimes sees spurious errors after interrupting
Removed CNRI copyright , changed notice to standard notice .
Merged revisions 83067 via svnmerge from
Quickly renamed .
merge 3 . 4 ( )
fix import
Patch : Prepend Modules / before Setup in stdout .
Adapt to modern times . . .
Fix spelling .
Updated the Mac documentation to the current state of affairs .
: document that default encoding is the same as locale . getpreferredencoding ( ) .
Issue : Fix refleak in fp_setreadl ( )
Issue : Call Tcl_ConditionFinalize for Tcl_Conditions that will
bpo - 38644 : Pass tstate to _Py_CheckFunctionResult ( ) ( GH - 17050 )
- img modules are no longer in PlugIns folder
Whitespace normalization .
merge 3 . 4
Fix signed / unsigned compiler warning .
Logical markup .
Issue : When falling back to the MIME ' name ' parameter , the
rename MathcObject to match object in doctrings for re module ( )
CObject use is marked as a Py3k warning , not a deprecation warning
SF Patch : Remove eval ( ) from csv
Merged revisions 78442 via svnmerge from
Add bytes in literal_eval doc .
Replace long with twodigits , to avoid depending
Closes : Merged fix from 3 . 3 .
- Install a symlink to the documentation ( which lives in the framework )
Remove a list comprehension , because a loop over the list
bpo - 40204 : Fix duplicates in the documentation ( GH - 21857 )
Issue : The preferred encoding is UTF - 8 on Android .
[ 693255 ] also back out corresponding NEWS item . . .
Use an event variable to wait for the server to be ready , rather than sleep ( 0 . 1 ) .
Doc : Fix missing negation . ( GH - 14640 )
Backport r67974 :
Use correct types for ASCII_CHAR_MASK integer constants .
fix argcount generation for arg lists containing tuple unpacks
Issue : Compile _math . c separately to avoid race condition
time and socket were already imported in the module , no need to re - import
Improve exception message for str . format ( GH - 12675 )
Issue : asyncio now closes sockets on errors
Bring Python into the next decade . ( GH - 17801 )
Convert test_userdict to use the collections . UserDict .
Fix comment typo .
merge
Issue : METH_KEYWORDS cannot be used alone in Python 3
Change \ u which is causing a syntax error . I ' m not sure if that should be , but
Issue 24316 : Fix types . coroutine ( ) to accept objects from Cython
merge 3 . 4
Revise the rules for building the distribution files for the HTML format .
First cut at a script to generate SGML / XML from an ESIS event stream ,
bpo - 36794 : Document that Lock . acquire is fair . ( GH - 13082 )
Usage nit : a bunch of " nor " s should have been a single " or " .
: HTMLParser can now parse correctly start tags that contain a bare / .
merge emphasized discouragement of overriding __import__
Python 3 . 10 . 0a3
Merged revisions 70723 via svnmerge from
Marc - Andre Lemburg < mal @ lemburg . com > :
Issue : Fix test_pause_reading ( ) of asyncio test_subprocess
adding a NEWS note for ( previously checked via the buildbot )
remove unused variable
config . h renamed to pyconfig . h
Temporary skip curses tests on non - tty ( issue ) .
bpo - 36710 : Add runtime variable to Py_FinalizeEx ( ) ( GH - 12937 )
Moved description of _urlopener adjacent to the related functions .
The letter headings must be requested explicitly with - - letters .
Closes : clarify message about bufsize changes in 3 . 2 . 4 and 3 . 3 . 1 .
Issue : Fix test_faulthandler on Android where raise ( ) exits with 0 ,
Add _close ( ) method that does the actual cleanup ( close ( ) asks the
markup fix
More robust way of choosing a non - existant uid ( faster , too ) .
whatsnew : Windows build uses ASLR and DEP .
Fix 13327 . Remove explicit None arguments from futimes , futimens , futimesat ,
Issue : Fix backup file creation in pindent . py on Windows
( Merge 3 . 4 ) Closes , : Use PY_FORMAT_SIZE_T instead of % zi or % zu
Issue : Fix rounding and implement _PyTime_ROUND_FLOOR in :
: add limits for command and data size to smtpd ; patch by Savio Sena .
Merged revisions 79388 via svnmerge from
Some really simple cgi examples . cgi3 is a MiniWiki .
intermediate
Issue : select . poll . poll ( ) again works with poll . POLLNVAL on AIX .
Users demand iterable input for named tuples . The author capitulates .
: mark refleak test as specific to CPython
Issue : Add one more unittest for CoroutineType . __await__
SF bug PyDict_UpdateFromSeq2 ( ) unused
changes for power ( * * ) operator
whitespace
dir ( ) method description revised based on comments from Greg Stein .
Added support for the horribly complex TimeRecord , so we can position
revert part of cefe4f38fa0e , as agreed with Éric
Issue : Respect the PYTHONCASEOK environment variable under
Issue 21069 : merge from 3 . 4
Explain that __init__ ( ) methods do not get to return values .
Issue : Don ' t clear environment in test_cleanup ( ) of test_builtin
PyString_FromFormatV , string_repr : document why these use sprintf
- Issue , : Fix os . execlp ( ) crash with missing 2nd argument .
: fix speling .
Tweak recipes and tests
Document PyNumber_ToBase
Added the actual builds of the carbon projects .
Looks like an oopsie - - I ignored a warning at my peril .
Remove duplicate sentence from alarm ( ) doc .
SF 1668596 / 1720897 : distutils now copies data files
Issue and Issue : silence output of setup . py when make is run with - s option .
In the _hashlib module , only initialize the static data for OpenSSL ' s
Adding tests for unittest command line handling of buffer , catchbreak and failfast .
Enhanced memory - reference information in the description of Py_BuildValue ( ) ,
Extra NEWS formatting fix for 3 . 6
merge
Removing the test of Element that causes ref - leak in GC ( issue ) .
Use PyString_CHECK_INTERNED .
Merged revisions 86676 via svnmerge from
. . . .
{ fulllineitems } is now an environment ; use it as such .
Issue : Support new buffer interface for struct . unpack and
bpo - 30317 : Fix multiprocessing test_timeout ( ) ( GH - 8621 )
Merge from 3 . 2
Issue : In the interpreter ' s initialization phase , unicode globals
- fix a GC bug caused by malloc ( ) failing
Merged revisions 70778 via svnmerge from
Fix the frozen bytecode for __hello__ ( betcha didn ' t know that existed
Rewritten - - this now tests the binascii * except * for the binhex
Document the two changes to the mailbox . py module :
s / the the / the / ( GH - 6287 )
Use METH_VARARGS instead of " 1 " in list method table .
PEP 227 implementation
Added new types to setattr ( ) as well . . .
Explain SQLite a bit more clearly
Code review of the new buffer protocol . Mostly add questions that should
Port to VS 2010 .
Fix typo in typing . py ( GH - 22121 )
Merged revisions 69316 via svnmerge from
SF bug 1442442 : LIST_APPEND optimization got lost in the AST merge .
Issue : Handle EINTR in _Py_open ( ) and _Py_fopen_obj ( )
Update to Expat 1 . 95 . 7 ; there are no changes to the Expat sources .
Initial revision
bail in unicode error ' s __str__ methods if the objects are not properly initialized ( closes )
: BaseCookie now parses ' secure ' and ' httponly ' flags .
Issue : Add test for fragment producing with XMLGenerator .
Minor cleanup
Issue : Distuguish call count and primitive call count in pstat output .
Try to make test_import a bit more robust
ensure the attribute name string is initalized before using it ( closes )
Use new file dialogs .
test_whitespace_eater_unicode ( ) : Make this test Python 2 . 1 compatible .
Patch : Expect ' ? ? ' in event fields . Fixes .
Allow using long integers as arguments to PyObject_GetItem ( ) , _SetItem ( ) ,
Basic documentation for the getpass module .
Added module synopsis for the chapter summary .
merge with 3 . 3
replace callable ( )
Fixed tests for issue ( pickling of IPv4Network was not tested ) .
Moved to Include and changed format to " < complete_version > "
Issue : Updated NEWS entry and added to logging documentation .
Combine the FastCall functions in cpickle .
Issue : update typing module documentation . ( merge 3 . 6 - > 3 . 7 )
: add provisional email policy with new header parsing and folding .
( match - string ) : Added a definition for older Emacsen .
PyUnicode_Join ( ) : Missed a spot where I intended a cast from size_t to
Fix logic error introduced in last commit . Also add a comment to explain what
Fixes platform . win32_ver on non - Windows platforms ( GH - 12912 )
CJK codecs : less magic macros , require explicit semicolon
Remove a useless XXX comment .
Issue : try to fix sporadic test_asyncio failure on FreeBSD 10 . 0
The compiler class for EMX was removed
SF patch , get python to link on OSF1 ( Dec Unix )
Update to VC 7 . 1 . Will backport to 2 . 4 .
In the description of enumerate ( ) , the indexing operators should not
don ' t rely on dict order
Improve docstring of list . sort ( GH - 8516 )
added a couple of missing periods .
Adjust to the new sysconfig regime : use ' get_config_vars ( ) ' instead
Fix for issue 6851 : urllib . urlopen crashes in a thread on OSX 10 . 6
Remove unnecessary uses of context in PyGetSetDef . See issue .
bpo - 34403 : On HP - UX , force ASCII for C locale ( GH - 8969 )
bpo - 36779 : time . tzname returns empty string on Windows if default cod … ( GH - 13073 )
Changed to use 8 - byte doubles . Also re - enabled optimizations , 68020
Merged revisions 80000 via svnmerge from
Separate Mac user interface modules into separate files , to better support
Issue : LockTests in test_imp should be skipped when thread is not available .
SyntaxError__str__ ( ) : Do more formatting of the exception here , rather
Issue : Plug refleak on SyntaxError in function parameters annotations .
Document fixed bugs
Put all the indexsubitem stuff together .
Issue : Don ' t assume in importlib . __init__ that __file__ is
Trailing whitespace is bad for . rst files .
Fix sporadic failure in test_startfile .
Fix trivial typo in test_dataclasses . py . ( GH - 5398 )
Updated logging HOWTO with a diagram .
Issue : Minor corrections to smtplib . ( Backport r60975 )
Fix compiler warning
Add reminder list of things to document ; mention sys . getcheckinterval ( ) and socket . timeout exception
Partial fix for SF bug 122780 ( msvcrt . locking constants aren ' t defined ) .
bpo - 40491 : Fix typo in syntax error for numeric literals ( GH - 19893 )
nonstandard_expectations is only added when in verbose mode , so we
Allow for 12 arguments max , in stead of 8 . Untested .
Merged revisions 72932 via svnmerge from
bpo - 37499 : Test various C calling conventions ( GH - 15776 )
Issue2469 : Correct a typo I introduced at r61793 : compilation error with UCS4 builds .
WTF is with this script ? It contained illegal syntax and illegal
fix issue : ftplib ' s storline method doesn ' t work with text files
Raymond didn ' t write any of the text this time , remove as author .
tabify
Issue : Re - implement imp . get_tag ( ) using sys . implementation .
1 . Clarify " tab / space " Error Dialog and " Tab Width " Dialog associated with
Issue : Refactor some code .
mwh @ sourceforge found that UnicodeError can be raised by compiling .
Issue : Fix module name in mock docs
Patch by Brian Gallew for DG / UX . I ' m not quite sure what it does but
Issue : Merge HTTPResponse fix from 3 . 5
bpo - 39984 : Pass tstate to _PyEval_SignalAsyncExc ( ) ( GH - 19049 )
Issue : Improve the misleading message from 2to3 when skipping optional fixers .
Normalize whitespace
Merge : : Handle non - string keys when generating ' fieldnames ' error .
Merged revisions 85003 via svnmerge from
Renamed module tkinter to _tkinter , moved source to _tkinter . c
asyncio doc : link create_connection ( ) to open_connection ( ) and create_server ( )
bpo - 36004 : Add date . fromisocalendar ( GH - 11888 )
Issue : Have test . test_importlib . builtin test both frozen and
[ Bug ] Fix typo in example
Merged revisions 81709 - 81710 via svnmerge from
flush before reading response from socket
Merge heads
fixed bugs in generic_setattr ( returned NULL instead of
bpo - 35810 : Incref heap - allocated types in PyObject_Init ( GH - 11661 )
Added prototype for PySys_SetArgv
* Increase test coverage .
Merged revisions 82842 via svnmerge from
Add IS_TRACKED and IS_MOVED macros . This makes the logic a little more clear .
bpo - 35031 : also disable TLS 1 . 3 for test_start_tls_server_1 on macOS ( GH - 14188 )
Added options that use square brackets in their names ; this ensures that
Split in a number of subtest for easy maintenance and continued Mac support .
More automated version replacement .
__validate ( ) : test for None - age , not not - ness
fix usage of undefined name ( )
The missing NEWS entry for r61542 . .
Issue : Fix socket module compilation on platforms with HAVE_BROKEN_POLL .
Document the build process more precisely . Suggested by Hugh Sasse .
Merged revisions 83407 via svnmerge from
- Don ' t ask for bring - to - front when quitting through an appleevent .
Update the python . gif icon for the Idle classbrowser and pathbowser
Add a Context Types section to parallel the Iterator Types section ( uses the same terminology as the 2 . 5a1 implementation )
Need to support single_input explicitly so from __future__ imports
merge
Merged revisions 74446 - 74449 via svnmerge from
- Fix comment macro in python . man
Issue : Correct and expand OS X deployment target tests in distutils
Implement appropriate __getnewargs__ for all immutable subclassable builtin
These files used to belong to the " sv " module , which was deleted .
[ Patch from Matthew Mueller ]
Fix synopsis .
In the example file name , use the current Python version , not a hardcoded
Update Moshe Zadka ' s email address to one works and might last a while .
Fix typo . Spotted by Bo Bayles on docs @ .
fix test_parser from tokenizer tweak
Issue : Avoid clock wrapping around in test_make_msgid_collisions .
Fix trivial typo in json module docstring ( GH - 2274 )
the usual
Added B format char to Py_BuildValue ( same as b , h , i , but makes
Patch : Ignore the locale ' s encoding if it is an empty string .
Setting textwrap . TextWrapper ( ) . expand_tabs to True calls expandtabs , not
Bump to 3 . 2 . 4rc1 .
Issue : Merge from 3 . 4 .
Add a subtract ( ) method to collections . Counter ( )
: fix indentation in example .
Added tag v3 . 4 . 3 for changeset b4cbecbc0781
Revise regrtest to compile entire standard library .
Bug : Deprecate passing floats to file . seek .
SF patch 103596 by Nick Mathewson : rause UnboundLocalError for
( Merge 3 . 1 ) Issue : Use sig_atomic_t type and volatile keyword in the
Issue : Document differences between synchronization primitives of
Logical markup . small nits .
Issue : Single var - keyword argument of dict subtype was passed
Deal with possible case of having time . tzname [ 1 ] containing UTC or GMT .
Issue : Docummented that ValueError is now raised instead of TypeError
Backport some main . c cleanup from the py3k branch
Give the profile module a module directive .
Adds _CRT_NON_CONFORMING_WCSTOK to retain two - parameter wcstok .
Closes : 8 space indent in tutorial
Binary flag is needed on windows .
Added help options to PDB
SF bug : calendar . weekheader not found in __all__
* Optimized list appends and pops by making fewer calls the underlying system
update links
Merge
Raise KeyError instead of OverflowError when getpwuid ' s argument is out of
: add a note about \ b .
added tests for the clean command
Added two # ! lines
Fix bug in class instance hash ( forgot to clear error condition ) .
Issue : Fix broken link in unittest . mock - examples . rst
Issue : Merge from 3 . 5
Add Misc / NEWS entry and relevant doc change for issue 10271 .
Revert merge from 2 . 3 alpha 1 release branch ; the change does not
Add Python porting howto from 3 . x .
Backport PEP 372 : OrderedDict ( )
Fix bug in test_tools that prevented building is separate directory .
: platform . architecture does not correctly escape argument to / usr / bin / file . Fix original patch
grant list . index ( ) a more informative error message
Update configparser docs for lowercasing rename .
: refactor poll - related classes ( )
merge
bpo - 1635741 , unicodedata : add ucd_type parameter to UCD_Check ( ) macro ( GH - 22328 )
: crtlicense . txt is displayed by the license ( ) command and should be kept ascii - only .
Added test for regression on SourceForge bug .
Fix asyncio tests on Windows : wait for the subprocess exit
Typo fix in docstring .
Fix closes issue12663 - Correcting the ArgumentParser . error description . Patch by Sandro Tosi .
int ( ) of a string is only expected to through ValueError , so do not use
bpo - 35482 : Fixes HTML escaping in CHM index and build location of NEWS file ( GH - 11224 )
Issue : Fixed unused variable warning on Unix .
Remove a couple of tp_xxx fields that you are not expected to
New class syntax .
MERGE : : platform . architecture does not correctly escape argument to / usr / bin / file . Fix original patch
Update to reflect the recent Grammar changes .
Fix markup in smtpd . rst .
bpo - 32710 : test_asyncio : test_sendfile reset policy ( GH - 11461 )
Indexing is a mess ; try to get the link targets closer to the relevant
For spawn * ( ) , point to the magic constants ; some readers have a hard
read ( ) method : clarify that strings are accepted and interpreted
merge 3 . 3 into 3 . 4
Rewrite to use test_support ' s fine fcmp instead - - I didn ' t know that
SF patch
Move importlib . abc . SourceLoader . source_to_code ( ) to InspectLoader .
String exceptions are gone and so are classic classes .
Oops , I forgot to include this file in the last commit ( 46046 ) :
Backport issue itertools fix from 3 . x .
asyncio / tests : Fix some ResourceWarnings
Merged revisions 79531 via svnmerge from
new style
Merge doc update
Fix test_ssl_presence when ssl is not present
Update version numbers .
bpo - 39314 : Closes parenthesis when autocompleting for functions that take no arguments ( GH - 20562 )
Make traceback objects collectable .
Don ' t die if win32api doesn ' t exist .
Merge : Fix test for existence of license URL .
Slight cleanup in distutils test_dist .
Merged revisions 67125 via svnmerge from
Merged revisions 76807 via svnmerge from
Issue : element_setattro returned incorrect error value .
Merge from 3 . 6 .
Logging documentation updates .
clarify buffer arg
Corrected citation markup in first paragraph .
Newly enabled test appears to leak :
Be more stylesheet friendly .
bpo - 30377 : Simplify handling of COMMENT and NL in tokenize . py ( )
Add a - address parameter to the LaTeX2HTML command line to direct
Use hex ( ) when outputting the various checksums so the test output is the
Make GenericAlias_Type and Generic_Type static . ( GH - 8076 )
Update pydoc topics .
The initial patch was not applied quite verbatim . This should one
Convert test_pkg to use unittest .
Fix some markup errors and adjust wording slightly .
Add " tkinter " to the list of directories to install .
Update permissions for Johannes Gijsbers .
mini - faq on porting python
Added getvar , getint , lower , tkraise = = lift to Misc class
Fix the date .
Issue : Change file system encoding on Windows to UTF - 8 ( PEP 529 )
fix formatting
: fix example in re doc .
include ( now ) int standard headers
Patch 1267 by Christian Heimes .
Optimistically , we label this as ready to go today .
merge with 3 . 2
Fix bug in islocked ( ) - - it would always return 0 .
new example that Matt posted to comp . lang . python
python - gdb . py : get C types at runtime
bpo - 35730 : IDLE - test squeezer reload ( ) by checking load_font ( ) ( GH - 11585 )
DO_TIMES must now be defined in the Makefile
Fix issue 10094 , by narrowing down the test for PROXY environment variables .
correct typo
Typo fix
Fix the NEWS Entry item . ( Thanks Stéphane Wirtel )
Remove the regrtest check that turns any ImportError into a skipped test .
As yet unused code for tool tips .
Suggested a whatsnew item
Fix merge in changeset e1dde980a92c .
Fixes IDLE crash on OSX : some versions of Tcl / Tk on OSX don ' t have a
Merged revisions 75889 via svnmerge from
bpo - 32604 : Improve subinterpreter tests . ( )
Issue 19509 : Don ' t call match_hostname ( ) twice in http . client .
Don ' t insist on cStringIO .
Issue : Add collections . ChainMap ( )
Use setUpClass ( ) and tearDownClass ( ) in test_multiprocessing .
Merged new pysqlite version 2 . 6 . 0 from trunk .
Issue : Update Misc / NEWS .
Issue : Fix PyErr_NormalizeException ( ) , handle PyObject_IsSubclass ( ) failure
PyUnicode_CopyCharacters ( ) fails if ' to ' has more than 1 reference
- sys . path [ 0 ] ( the directory from which the script is loaded ) is now
: check for the availability of the thread module .
Issue : Merge from 3 . 6
[ Patch ] Test the ' nil ' extension
Merge test from 3 . 2 .
Add another link to colorsys docs .
: fix os . popen * regression from 2 . 5 : don ' t execute commands as a sequence
Add lzma . { encode , decode } _filter_properties ( ) .
map ( None , . . . ) is not supported in 3 . 0 .
plug refleak
Lock methods acquire ( ) and locked ( ) now return bools .
Issue : Fix code formatting in datamodel . rst .
Mods by Tony Lownds ( patch 490100 , slightly massaged by me ) to make Tkinter
Initial revision
Issue : Proxy cr_await and gi_yieldfrom in @ types . coroutine
bpo - 4963 : Fix for initialization and non - deterministic behavior issues in mimetypes ( GH - 3062 )
Bug : rename variable in heapq doc example , to avoid shadowing
Issue : bytes constructor now raises TypeError when encoding or errors
Accept keyword arguments for __import__ and doc the addition of the level param from PEP 328 .
Charles G Waldman :
Make test_unicode pass after the lexer was fixed to turn unicode errors
merge with 3 . 5
Fix a few small typos in the docstrings .
Explain the strategy to avoid huge alignment shifts in _mpd_qadd ( ) in detail .
Forward port r70471 : Add object_pairs_hook . Issue 5381 .
Avoid call_function_tail ( ) for empty format str
Fix refcount leak in PyUnicode_EncodeCharmap ( ) . The bug surfaces
bpo - 1635741 : Port _lsprof extension to multi - phase init ( PEP 489 ) ( GH - 22220 )
Merged revisions 81862 via svnmerge from
Add x64 - temp to ignore , prepend a forward slash to " build / " to include
Correct attribute name .
Changed warn ( ) to warning ( ) in logging tests .
Merged revisions 86528 via svnmerge from
Add PyType_GetSlot to python3 def file . Patch courtesy Martin v . Löwis .
Close : Fix extended attributes tests in test_os for SELinux
Add fast paths to deque_init ( ) for the common cases
Regenerate for glibc 2 . 2 . 3 .
SF : fix so hyphenated words surrounded by punctuation are
Issue : Remove unneeded mkstemp helper in test_subprocess
Issue : Introduce a new scheme for computing hashes of numbers
minor . rst fix
In the Swedish alphabet , Å is sorted after Z
Merge from 3 . 6
Trivial cleanup : remove redundant variable stores in ceval . c ( )
issue1177 : Ported Facundo ' s from urllib2 to urllib , accepting 2xx responses .
Merged revisions 82433 via svnmerge from
Clean up tests by reusing functions from other modules :
Patch : Add documentation for xml . etree .
Patch : Update VC6 to use current PCbuild settings .
Reword the text on the demise of __dynamic__ somewhat , correcting a
Issue : Added the escape ( ) function to the glob module .
Text . __init__ ( ) : Make sure the data parameter is a string ( 8 - bit or
Fix a markup error in socket . rst
Issue : Clarified documentation for TimedRotatingFileHandler weekday rotation .
Commit __version__ change .
Revert r80963 - it broke compilation everywhere
Improve error message handling .
Store default startup options in preference file or
Another fix for Windows GUI apps
Added descriptions of EnvironmentError and OSError ; updated IOError to
Windows build : close out 2 . 2 , prep for 2 . 3 .
De - duplicate contents of pytime . h
bpo - 34195 : Fix case - sensitive comparison in test_nt_helpers ( GH - 8448 )
Fixed typos in TZ format description
No longer relevant .
docs : macos - change " versiona " to " versions " ( GH - 18467 )
Revert r88639 ( the optimization changes behaviour and breaks buildbots )
Issue : GzipFile now raises EOFError when reading a corrupted file
remove a versionadded
Add news about dictionary ( ) constructor .
: FAQ entry for list mutation . ( See also 6375bf34fff6 . )
Fix for documentation bug . Fixes issue 3904 .
format_obj : make it static
deprecate some useless , noop methods in symtable
Remove ' const ' from local variable declaration in string_zfill ( ) - - it
main ( ) : Include a \ label { modindex } in the output so latex won ' t let us re - use
Complete the integration of Sam Bayer ' s fixes .
Improve and expand identity tests .
Move history to HISTORY . txt
Merged revisions 61038 , 61042 - 61045 , 61047 , 61050 , 61053 , 61055 - 61056 , 61061 - 61062 , 61066 , 61068 , 61070 , 61083 , 61085 , 61092 - 61097 , 61103 - 61104 , 61110 - 61112 , 61114 - 61115 , 61117 - 61125 via svnmerge from
Make fnmatch be more PEP 8 compliant .
: avoid shadowing built - in map in mmap examples . Initial patch by Aman Shah .
Change % s argument for PyUnicode_FromFormat to
Issue : Correct characters in TextWrapper . replace_whitespace docs .
Use the proper autoconf macro for checking byte order .
Add const qualifier to PyErr_SetFromErrnoWithFilename and to
Issue : Validate ellipsis tokens in relative imports .
time_strptime ( ) uses PyObject_Call ( )
bpo - 34011 : Update code copying DLLs and init . tcl into venvs . ( GH - 8253 )
Remove UTF - 8 BOMs .
cleaned up the module ( PEP 8 + old fashion test removal )
This patch enables Cygwin Python to build _tkinter against Tcl / Tk 8 . 4 .
Issue : The null character now correctly passed from Tcl to Python .
bpo - 30950 : Convert round ( ) to Argument Clinic . ( )
fix wrong credit and issue id given in previous commit
Merged revisions 82595 via svnmerge from
Retouch my last change after a comment on style from Guido .
Merged revisions 77020 via svnmerge from
merge from 3 . 2
Issue : improve links in new distribution docs
SF , fix memoryview ( ' test ' ) so it works in debug mode too .
Remove assumption that cls is a subclass of dict .
Use Boolean values for the capturestderr flag .
Merge with 3 . 4
note that the alarm function is not available on Windows .
Issue : Fixed formatting Windows error messages on Wine .
Whitespace normalization and fold some long lines .
Drop 0 parameter
Merge issue : Prevent infinite recursion for ABC Set class operations .
Provide encoding as keyword argument ; soften warning paragraph about encodings
Push a little more here to avoid style information in the generation
When the HTML output dir is specified using - - dir = ~ / foo , the tilde
Fix deprecation warnings in test_long_future . py
Let the default encodings search function lookup aliases before trying the codec import . This allows applications to install codecs which override ( non - special - cased ) builtin codecs .
Make mailing list thread reference an inline link , reword i . e . abbreviation .
Update Misc / NEWS for issues and . ( And , technically , . )
Update importlib . h
Issue : adds a more solid test suite for asyncore
Fix SF bug , test_posix fails : getlogin
bpo - 20104 : Add flag capabilities to posix_spawn ( GH - 6693 )
Extend function ( ) to support an optional closure argument .
Add launcher program
Document PyType_CheckExact ( ) , PyType_IS_GC ( ) .
- Issue : Update the bsddb module to pybsddb 5 . 3 . 0 , supporting
Merged revisions 80991 via svnmerge from
Change test_htmlparser to reflect the HTMLParser - > html . parser
Issue 19276 : Fix tests for wave files on big - endian platforms .
bpo - 32323 : urllib . parse . urlsplit ( ) must not lowercase ( ) IPv6 scope value ( )
Issue : try to fix sporadic failures in test_ftplib
Add the directories where DarwinPorts installs library and include files to
This branch is for 1 . 6 & forward .
Charles Waldman writes :
( py - indent - line ) : take an optional universal argument which says to
: return value of flag from Event . wait ( ) . OKed by Guido .
Issue : Revert commits in 3 . 4 branch which is security - only fixes .
: Add a tolerant mode to HTMLParser .
Issue : fix Py_IS_INFINITY macro to work correctly on x87 FPUs .
: merge with 3 . 2 .
\ seemodule { } : Use logical markup for the module name , to ensure it picks up
bpo - 35113 : Fix inspect . getsource to return correct source for inner classes ( )
Issue : on 64bit platforms , str . decode fails on very long strings .
Print a more meaningful message when a test ' s output file wasn ' t
SF Patch : miscellaneous doc typos
bpo - 29751 : add Cheryl Sabella to Misc / ACKS ( GH - 1268 )
Issue : Fix an exception when a " dummy " thread is in the threading module ' s active list after a fork ( ) .
keybinding configuration
Merged revisions 84754 - 84755 via svnmerge from
PyFunction_Call ( ) did not check the result of PyObject_Repr ( ) for NULL , and
bpo - 25054 , bpo - 1647489 : Added support of splitting on zerowidth patterns . ( )
Fix minor cut - and - paste typo .
asyncio : sync with Tulip
bpo - 38328 : Speed up the creation time of constant list and set display . ( GH - 17114 )
Close issue23900 : add default __doc__ to new enumerations that do not specify one .
Issue : Add unit tests on os . spawn * ( )
Added internal routine PyString_Fini ( ) which deletes all interned
Fix SF bug . Handle string exceptions even if unraisable ( ie in __del__ ) .
Merge with 3 . 2
Merge heads
bpo - 20260 : Implement non - bitwise unsigned int converters for Argument Clinic . ( GH - 8434 )
bpo - 1635741 : Port syslog module to multiphase initialization ( GH - 19907 )
Michael Hudson < mwh21 @ cam . ac . uk > :
: dont raise an unexpected IndexError if a voidresp ( ) call has an empty response .
longobject . c : add an assertion to ensure that MEDIUM_VALUE ( ) is only called on
Updated reprlib ' s documentation to mention the renaming .
Add option for no updates on typing
: PYTHONPATH delimiter is os . pathsep .
Got the new structure working with MSVC 4 . 2 .
Quick ' n dirty hack : Increase the magic by 2 to force a rebuild of pyc / pyo files on the build bots
news entry for r73825
Revert r83395 , it introduces test failures and is not necessary anyway since we now have to nul - terminate the string anyway .
Get mock coverage back to 100 % ( GH - 18228 )
Merged revisions 88610 via svnmerge from
Markup fix
Another name ; should be added to Python 2 . 2 . 1 .
Fixed : backported logging fix from 2 . 7 .
Merge with hg . python . org .
Merged revisions 69924 , 69998 - 69999 via svnmerge from
Merge 3 . 3 .
Rewritten keys ( ) and has_key ( ) to avoid memory leaks .
- Changed GestaltEqu . h to Gestalt . h
: s / assertRaisesRegexp / assertRaisesRegex / g
Issue . Fixed a potential FD leak in subprocess . py .
Patch : Persistent connections in BaseHTTPServer .
Added ( minimal ) StringIO docs .
Issue : Fix zlib . compressobj ( ) to handle PyThread_allocate_lock ( ) failure
Remove legacy support for the BrowserControl module ; the webbrowser
Blacklisting LSInit and LSTerm , which are deprecated . Partial fix for 853558 .
Remove reference to the " t " open ( ) mode as it is platform dependent .
update notes to talk about - X showrefcount
Issue : Update . gitignore to ignore Windows build output .
Document the repr module . This removes the entire " Code objects and
Merged revisions 77130 via svnmerge from
Closing patch - - After everyone agreed .
Syntax highlighting only works when > > > lines are accompanied by . . . lines
Merged revisions 82828 via svnmerge from
remove unused argument ( closes )
Always room to remove more formatting from the HTML !
Comment repair .
Issue : Improve message in configparser . InterpolationMissingOptionError .
Contributed updates from Harald Hanche - Olsen , giving details of the branch
Delete the bytecode from the library and tests before running the
Got rid of 68K support .
[ Bug ] When you use asyncore with a non - default map , methods
Issue : datetime . astimezone ( ) method will now supply a class
Issue : Rewrite test in main thread and avoid race condition
Add Macros defining new names for all external symbols .
_PyMalloc_ { Malloc , Realloc } : Strive to meet the doc ' s promises about
Issue : disable faulthandler timeout thread on all platforms
Merged revisions 81522 via svnmerge from
Issue : Fix " z # " format of PyArg_Parse * ( ) function : the size was not
Add support for FreeBSD 6 .
Fixes Issue : http . cookiejar was not properly handling URLs with a / in
: Add news item .
Issue : Fixed possible crashes when unpickle itertools objects from
SF bug : Typo about PyErr_WriteUnraisable ( )
The interruptRoutine attribute is gone under Carbon . Luckily it appears that nothing used it .
Make the pdb displayhook compatible with the standard displayhook : do not print Nones . Add a test for that .
merge : make Header and make_header handle binary unknown - 8bit input
Re - apply distutils2 changes lost before the merge of packaging .
Merged revisions 87575 via svnmerge from
Add another news entry .
Issue : Fix reference leaks in error paths in dictobject . c .
Temporarily revert 59967 until GC can be added .
Bug
Issue : Fix a ctypes problem setting bitfields more than 31 bits
Bug : clarified that ` doctest ` does not support examples
Fixed spacing and unbalanced brackets or parenthesis .
Issue : Remove executable bits from C source files and several forgotten
Always close files in distutils code and tests ( ) .
Bypass set specific optimizations for set and frozenset subclasses .
Remove direct call ' s to file ' s constructor and replace them with calls to
Change tb_here ( ) prototype .
bpo - 33556 : Remove reference to thread module from docstring ( GH - 6963 )
: add reference to ioctl in fcntl module for platforms other than Windows .
Issue : Set TERM = ' ' by default in assert_python_ *
Fix some broken / missing function : : declarations in turtle docs
Test that True can be copied .
Issue : Fix ResourceWarnings in test_ssl .
Merged revisions 85982 via svnmerge from
SF patch : __main__ for whichdb . py
Added a couple more requests from users .
Merge heads
Set this test up so that we don ' t have to create xx / LC_MESSAGES in the
bpo - 37972 : unittest . mock . _Call now passes on __getitem__ to the __getattr__ chaining so that call ( ) can be subscriptable ( GH - 15565 )
Merged revisions 88465 via svnmerge from
Whitespace normalization
Merge with 3 . 3
* regression bug , count_next was coercing a Py_ssize_t to an unsigned Py_size_t
Userlist . copy ( ) wasn ' t returning a UserList .
Hm . Add Unix binding for replace , too .
Added rindex ( ) . index ( ) and rindex ( ) interpret negative start index
SF 643115 : Set . _update ( ) had a special case for dictionaries which allowed
add references between getopt and optparse docs
Issue 11715 : Build extension modules on multiarch Debian and Ubuntu by
Issue : Fix os . statvfs ( ) , handle errors
Added note about the pyexpat memory leak fix .
Added descriptions for some modules that previously did not have any
Merge with 3 . 4 : Issue : tarfile now raises a ReadError if an archive is truncated inside a data segment .
Issue : threading : Fix a race condition in Event . wait ( ) that made it
: email versions have gotten out of sync , 2 . 7 is actually 4 . 0 . 3 .
Issue : Fixed the documentation and added tests for PyUnicode_AsUCS4 ( ) .
bpo - 38787 : Update structures . rst docs ( PEP 573 ) ( GH - 19980 )
Issue : Add a __sizeof__ implementation for BytesIO objects .
Really do replacement of & amp ; last to avoid bad interactions between & amp ;
Issue : Merge subprocess fix from 3 . 4 into 3 . 5
move test outside WITH_THREAD section
Add missing check of PyDict_Update ( ) ' s return value in _elementtree . c
Issue : Fix http . server to again handle scripts in CGI subdirectories ,
Improve docs for super ( ) .
Issue and :
bpo - 30928 : Update idlelib / NEWS . txt to 2017 Dec 3 . ( )
Added recognition of ' l ' or ' L ' as long integer suffix
merge 3 . 1
Added rationale for defaulting to 1 in the functional API .
Split - out a fourth section in the descriptor HowTo guide ( GH - 22965 )
Add item
Oops , must convert hyphens to underscores in keys of aliases dict .
Back out 7e9605697dfc , 2e3c31ab586a , 759b2cecc289 .
Don ' t use printobject ( ) to print a string ( filename ) .
bpo - 18578 : Rename and document test . bytecode_helper as test . support . bytecode_helper ( GH - 15168 )
bpo - 1635741 port _curses_panel to multi - phase init ( PEP 489 ) ( GH - 21986 )
Enabled resID input / output parameter to NewMovieFromFile . This changes
: merge with 3 . 3 .
Merged revisions 78340 via svnmerge from
Test the applesingle decoder .
Avoid title case in flowing text ( backport from 3 . 2 )
Patch by Dean Draayer : support for indeterminate progress bars . You
Remove last remnants of the ipaddr package .
use only the hex version , since the revno is unreliable across repos
Issue 10025 : Let docs for random . seed ( ) describe what it does , not how it does it .
bpo - 42435 : Speed up comparison of bytes and bytearray object ( GH - - 23461 )
_Py prefix is verboten for static entry points
Add compiler test regarding optional arguments .
SF patch : file and cStringIO raise a ValueError when next ( ) is called
Fixed : - - formats = tar , gztar was not working properly in the sdist command
Add news about FutureWarning and PEP 237 stage B0 .
bpo - 30687 : Fixes build scripts to find msbuild . exe and stop relying on vcvarsall . bat ( )
Merged revisions 87442 via svnmerge from
Issue : memoryview . cast ( ) is now allowed on zero - length views .
Issue : Fixed potential crash in PyUnicode_AsDecodedObject ( ) in debug build .
Add some additional cleanup transformations .
Correct typo in AddressList . __getitem__ . By Moshe Zadka .
Remove the conditional for setting query . ( )
Issue : Minimally document PyFrameObject
kill extra word
Fix wording in NEWS entry .
Rewrap long lines in Misc / NEWS .
* Mass change : get rid of all init ( ) methods , in favor of __init__ ( )
Issue : Merge XML - RPC client fix from 3 . 5
Merged documentation update from 3 . 2 .
Fix a few more memory leaks
Issue : Update decimal . __all__
post release cleanup
Updated MSVC project files to follow PyCObject removal ( r73729 )
_asyncio uses _PyObject_CallMethodIdObjArgs ( )
Merge 3 . 2 .
[ Patch ] zlib . crc32 doesn ' t handle 0xffffffff seed . Add tests and bugfix . Bug reported by John Schmidt ; bugfix by Danny Yoo .
bpo - 6761 : Enhance __call__ documentation ( GH - 7987 )
Guess tab size on the Mac from tab resource .
final patches from Neil Schemenauer for garbage collection
Merge : : Fix typo in example .
Finn Bock ( SF patch ) : Avoid outdated exec form in
rd_object ( ) with exception is fatal error
typo fix
Merged revisions 75982 via svnmerge from
_pickle : Fix load_counted_tuple ( ) , use Py_ssize_t for size
Check for potential error returned by _ungetch ( ) .
Write out " phi " consistently .
Issue 24017 : Drop getawaitablefunc and friends in favor of unaryfunc .
Issue 24017 : Fix asyncio . CoroWrapper to support ' async def ' coroutines
replace calls to get the initial values with the raw constants .
Create a TextMate directory in Misc to house a Python - Dev bundle .
Merge : : Preemptively fix some other possible timing issues .
Issue : Remove obsolete support for view = = NULL in bytesiobuf_getbuffer ( )
Add soft keywords to the documentation ( GH - 21185 )
Issue : Handle all enumerations in _Py_ANNOTATE_MEMORY_ORDER
Add a missing forward declaration for PyFileIO_Type . The _fileio module now compiles with - - with - pydebug .
Merge : : make is_attachment a method .
Fixed to use ' reinitialize_command ( ) ' to fetch " install " and " install_lib "
Issue : Speedup some comparisons . This also removes the last call
bpo - 42131 : Add PEP 451 - related methods to zipimport ( GH - 23187 )
Use abort ( ) rather than exit ( ) to appease tools like rpmlint . abort ( ) is used
Revert previous checkin .
Fix typo in Lib / concurrent / futures / thread . py ( GH - 13953 )
Change security policy - - only accept requests from current host .
Fixed index entries to be more consistent .
Fix markup error and minor consistency nit .
DOC : attribute PyPy for the idea behind LOAD_ATTR cache ( GH - 23036 )
Added URLs for zlib ' s home page and the Windows DLL ' s home page .
merge with 3 . 6
use error label instead of breaking eval loop ( closes )
bpo - 33089 : Multidimensional math . hypot ( ) ( GH - 8474 )
Issue 8316 : make test_gdb robust in the face of differing terminal
Issue , issue : fix issue number of commit 2c0da1c4f063 .
More markup additions
Add TODO item
Updated , mappingobject - > dictobject plus some other minor things .
Drop reference to pythoncore , to avoid linking python34 . dll
Open files in binary mode to avoid newlines transformation .
Merged revisions 63438 via svnmerge from
closes bpo - 34646 : Remove PyAPI_ * macros from declarations . ( GH - 9218 )
cPickle , version 0 . 1 .
Add getmodulename ( ) and getmoduleinfo ( ) routines to inspect filenames .
bpo - 32124 : Document C functions safe before init ( )
Remove the Py_FatalError ( ) from initpyexpat ( ) ; the Guido has decreed
Remove compatibility code for Python < 2 . 1 , < 2 . 2 and < 2 . 4 .
Tim ' s changes ; removed some remaining non - functional ifdefs
: Unwrap fp in BytesParser so the file isn ' t unexpectedly closed .
Remove outdated include ; this include was breaking OS X builds using
Fix typo .
Issue : Trying to fix test_use on Windows .
Bug : socketmodule now builds on Sun platforms even when
Revert 76452b892838 as per http : / / mail . python . org / pipermail / python - dev / 2011 - July / 112243 . html .
Issue : The test_find_mac test in test_uuid is now skipped if the
SF patch : fix for bug 639806 : default for dict . pop
Re - licence statistics . py under the standard Python licence .
bpo - 37160 : Thread native ID NetBSD support ( GH - 13835 )
Clarify the relationship between indices .
bpo - 34007 : Skip traceback tests if the Program Counter is not available . ( GH - 9018 )
Merged revisions 78830 via svnmerge from
Lance has converted his modules to the new naming conventions , and
Fix deprecation warnings in test_cgi . py
Add Shane .
Convert file . readinto ( ) to stop using METH_OLDARGS & PyArg_Parse .
Merged revisions 67171 via svnmerge from
Better error checking in initmath ( ) .
Fix bug in _insert_thousands_sep : too much zero padding could be
Fix typo reported by Sandro Tosi .
Issue : Added test for venv prefixes .
Move PyObject_Malloc and PyObject_Free here from object . c . Remove
Add comment about removal of CoreServices / CoreFoundation compilation against
Add minor markup for a string .
: news entry .
Merged revisions 73138 via svnmerge from
Move a news entry to the right section ( + light reformatting )
Marc - Andre Lemburg < mal @ lemburg . com > :
Issue : Fix wrong name in idlelib / tabbedpages . py .
Changes to make these work under OSX as the main program for a
Issue : Suppress warning in obmalloc when size_t is
Issue : A number of two - argument Decimal methods were failing to
Issue 4910 , patch 2 of ( probably ) 3 : pave the way for renaming of
( Merge 3 . 2 ) Issue : PyOS_StdioReadline ( ) retries fgets ( ) if it was
Fixed signed / unsigned comparison warning
- apply patch from issue 7005
Merge from 3 . 3
Great renaming .
Patch : Exclude ctypes package in Win64 MSI file .
Peter Schneider - Kamp :
Document complex ( ) with string arg .
add Allison Kaptur ( )
* drop the unreasonable list invariant that ob_item should never come back
issue10883 - Silence some ftp related ResourceWarnings in test_urllib2net . Patch by Nadeem Vawda .
Synchronize maintainers file
SF # 585913 , Adds Galeon support to webbrowser . py
SF bug Win32 : pydoc command isn ' t executable .
Issue : don ' t use PY_SIZE_MAX for overflow checking in List_New . Patch by Xiang Zhang .
Fix minor typos
Don ' t load the config . h file , even under Unix , because we never use the
bpo - 13802 : Use non - Latin characters in IDLE ' s Font settings sample . ( )
Use more dict2 functions . /
Issue : logging : Extra checks on the " level " argument in more places .
Fix some minor nits about the use of \ optional in parameter lists .
" make clobber " should remove modindex . html as well .
SF patch ( by Christos Georgiou )
Issue4335 : Added a test for inspect . getsourcelines with a module without EOL at EOF .
If _stat_float_times is false , we will try to INCREF ival which could be NULL .
Additions for Mark Hammond ' s Win32 specific hacks .
Issue : backport of 3 . 2 ' s asyncore / test_asyncore to 2 . 7 .
Fix minor formatting nits and remove unnecessary comment .
Fix typo in asyncio . AbstractServer documentation
get_sourcefile ( ) : use PyUnicode_READ ( ) to avoid the creation of a temporary
Fix bsddb . db . DBError derived exceptions so they can be unpickled .
Fix broken API descriptions in comments .
Merged revisions 87791 via svnmerge from
Remove Lib / _sysconfigdata . py from . gitignore
* Note list optimizations
bpo - 30908 : Fix dangling thread in test_os . TestSendfile ( )
provide a fallback for xxmodule . c in case the buildir is not present
Fix for catestrophic errors in previous checkin ( Argument Clinic rollup patch ) .
Add info about Windows filesystem limits .
Fix dropped word .
Merge 3 . 4 ( asyncio changes )
Fix whitespace
Issue : Document pkgutil . get_data ( ) doesn ' t work with namespace packages .
Add skipping to struct test that only applies when overflow masking is in effect
.
Merged trailing whitespace fix .
Issue : use short float repr where possible .
Remove sorting HOWTO , after converting it to a wiki page at http : / / wiki . python . org / moin / HowTo / Sorting
Add two tests for string . zfill
Use Py_ssize_t in helper function between Py_ssize_t - using functions .
Plug the most annoying recursive printing problem - - reset ' _ ' to None
fix ftplib resource warnings
: document that the parser only accepts \ \ n newlines .
Merged revisions 81100 via svnmerge from
bpo - 33906 : Rename idlelib . windows as window ( )
What ' s New in Python 3 . 4 : more security changes
Fixed socket_gethostname ( ) on windows .
merge with 3 . 2
Issue : bytes . fromhex ( ) and bytearray . fromhex ( ) now ignore all ASCII
Added binascii
Followup to 862ab99ab570 : I forgot to add the magnificent test_frame . py .
merge from 3 . 2
Still not a real test , but at least it runs again
Patch : Return absolute path for mkstemp . Fixes .
Fix up an awkward sentence , pointed out by Chris Ryland < cpr @ emsoftware . com > .
whatsnew / 3 . 5 : Reformat PEP 475 to render in less space ; add " ugly " ex in 465
Fix test failures ( - - without - doc - strings ) .
Issue : Special case print & exec syntax errors
Issue 7244 : Fix indentation in C code . Fix test to not sent output to stdout .
Fix signed / unsigned mismatch .
: allow easier creation of AST nodes .
Several Windows related cleanups :
Fix typo in example ( should be backported , but my maintenance branch is woefully out of date )
Avoid a couple of " value computed is not used " warnings from gcc - Wall ;
bpo - 41810 : Reintroduce ` types . EllipsisType ` , ` . NoneType ` & ` . NotImplementedType ` ( GH - 22336 )
Move some \ index entries around so we can avoid using empty comments ;
bpo - 36051 : Drop GIL during large bytes . join ( ) ( GH - 17757 )
NEWS : issue is related to Library ( sys ) , not Core and Builtins
Final touch . . .
Refactor code for translating " power " nodes .
SF bug : pdb imports ' repr ' , causing name collision
: add a test .
Fix a typo and remove some unneeded markup
Fix spelling .
Merged revisions 80282 via svnmerge from
Call PyErr_NoMemory ( ) when PyMem_Malloc ( ) fails .
Branch merge
Merged revisions 68499 via svnmerge from
Backport of r74435 . Not merged / blocked w / svnmerge . py as the tool is erroring out on me .
merge 3 . 4 ( )
Patch : Deal with emptied dumbdbm files correctly .
Issue : Fixed copying object with state with boolean value is false .
bpo - 35887 : Add make regen - importlib step to importlib . _bootstrap docstring ( GH - 11777 )
Bug report and partial patch from Michael Deegan < michael @ ucc . gu . uwa . edu . au > :
Minor doc fix in Clinic howto .
Issue : Plug ref leak .
bpo - 38234 : Cleanup getpath . c ( GH - 16367 )
Rewrote the docs for supporting cyclic garbage collection to reflect
Merge 3 . 6 , fix for
Issue : Complete attribute names even if they are not yet created
Handling of defaultLocation argument was wrong , causing a crash when used . Fixed .
Clarify " cardinality of " as " number of elements in " as many readers do
bpo - 40910 : Export Py_GetArgcArgv ( ) function ( GH - 20721 )
: fix code highlight in multiprocessing examples . Patch by Tshepang Lekhonkhobe .
Merge with current default
Bug : Add classes LocaleTextCalendar and LocaleHTMLCalendar ,
merge 3 . 4 ( )
merge 3 . 4 ( )
Fix sentence stating TextEdit only saves in RTF ; can save in plaintext as well .
Merged revisions 79311 via svnmerge from
Bug : the Mac filesystem does have accurate information
Attempt fix of by refining setup / teardown logic .
Update argparse docs to follow order of ArgumentParser ( ) arguments .
Merged revisions 80314 - 80315 via svnmerge from
: xml . etree . ElementTree now serializes correctly the empty HTML elements " meta " and " param " .
bpo - 27181 : Add statistics . geometric_mean ( ) ( GH - 12638 )
Issue 6003 : ZipFile . writestr " compression_type " argument
Issue : Test temporarily skipped while failure investigated .
bpo - 41995 : Fix null ptr deref in tracemalloc_copy_trace ( ) ( GH - 22660 )
bpo - 30215 : Make re . compile ( ) locale agnostic . ( )
Fix :
Closes : small clarification in design FAQ .
merge with 3 . 3
More typo fixes .
A much revised version of SF patch 514662 , by Naofumi Honda . This
add another advancement
logging : Added _logRecordClass , getLogRecordClass , setLogRecordClass to increase flexibility of LogRecord creation .
Issue : add long option names to unittest discovery docs .
Removes use of ValueOrDefault function that is not always available on the buildbots .
Issue : Fix build of parsermodule under Cygwin .
Don ' t use the fncychap package for HOWTO documents ; appendices break . Still
save_bool ( ) : simplified .
stop using Py_FindMethod
fix reference to File Object documentation
Initial revision
Issue : Extend the PyModuleDef_HEAD_INIT macro to explicitly
post - release
Update auto - detection for Konqueror to include KDE 2 - - the kfm command is
Relocate importlib . _case_ok to importlib . _bootstrap .
bpo - 40417 : Fix deprecation warning in PyImport_ReloadModule ( GH - 19750 )
Issue : Use binary prefixes ( KiB , MiB , GiB ) for memory units .
adjust formatting
Fix compilation when SSL_OP_SINGLE_ECDH_USE isn ' t defined
Added tbd ' s on python - font - lock - keywords
Fix a typo . ( closes )
Explain the use of charset parameter with Content - Type header . Issue11082
Removed all the mixed indentation face stuff . It ' s not as helpful as
bpo - 36266 : Add module name in ImportError when DLL not found on Windows ( GH - 15180 )
Add a note for strptime that just because strftime supports some extra
some re - design
Issue : xxlimited . c errors when building 32 and 64 bit on Windows
Fix symbol table pass to generation SyntaxError exceptions that
Add try - finally around main loop .
Issue : More tests ; fix nits in compiler . c
Issue : The " encodingname " part of PYTHONIOENCODING is now optional .
Issue : Use more " bytes - like object " throughout the docs and comments .
cleaned up the bdist_dumb module
Issue : Allow Decimal constructor to accept non - European decimal digits , as recommended by the specification .
Remove some dead code ( PyChecker )
Merged revisions 80166 via svnmerge from
write a short little section for multiprocessing ; it still needs help
Issue : Fixed buffer overread in BytesIO . readline ( ) if a position is
Remove unused code .
Fix spelling error and remove Windows line endings .
Revert " bpo - 22087 : Fix Policy . get_event_loop ( ) to detect fork ( GH - 7208 ) " ( )
asyncio , Tulip issue 171 : BaseEventLoop . close ( ) now raises an exception if the
C API docs : Py_IsInitialized is always safe to call ( GH - 12630 )
Closes : Add support for text modes to gzip . open ( ) .
Merged revisions 74493 via svnmerge from
Work around latex2html problem ; reported by Gerrit Holl
undefine SET_SYS_FROM_STRING_BORROW after its done being used ( closes )
Have distutils . sysconfig close a file to remove ResourceWarnings coming up
BZ2Comp_compress ( ) : Explicitly cast the LONG_LONG size argument to
bpo - 22602 : Raise an exception in the UTF - 7 decoder for ill - formed sequences starting with " + " . ( GH - 8741 )
bpo - 36520 : Email header folded incorrectly ( )
whitespace fix
Merged revisions 83030 via svnmerge from
: add note about fread ( ) .
Issue : Fixed a typo in PyErr_Fetch example .
_reconstructor ( ) : there ' s no need for tricks with assignment to
Merged revisions 81533 via svnmerge from
Have importlib look for pre - existing path separators when joining
Remove trailing spaces
Issue : Fixed copying object with state with boolean value is false .
bpo - 39185 Add the d [ etailed ] and q [ uiet ] verbosity levels for msbuild ( GH - 17791 )
: Merge with 3 . 1 .
Allow configure to detect whether ndbm . h or gdbm / ndbm . h is installed .
Issue : Adjust f - strings doc for interable unpacking
Closes : Merge with 3 . 4
Update PEP292 section
merge 3 . 3
Fixup new itertools recipes .
bpo - 39073 : validate Address parts to disallow CRLF ( )
Copying the email package back , despite its failings .
Issue20467 : clarify __init__ ' s role
Add some missing news . Add some periods .
Removed deprecated method from pstats .
Removed test from test_import . It will never work on all systems , see
Merge heads
Added cwd ( )
Markup fix
We no longer check for libsun . a .
Remove our implementation of memmove ( ) and strerror ( ) ; both are in the C89
merge heads
Added Objects / capsule . c to project files .
Update PY_VERSION to indicate that we ' re beyond 2 . 3a2 now .
Added SIZEOF_VOID_P and HAVE_LONG_LONG
Patch : Make terminal become controlling in pty . fork ( ) .
Fix issue references .
Optimize _Py_strhex_impl ( ) ( GH - 19535 )
Issue : Another performance optimization for BZ2File .
Issue : Fix compilation on Windows
Beautify code examples .
fdconv ( ) : Do not second guess the error condition returned by
Issue : fix calling accept ( ) on a SSLSocket created through SSLContext . wrap_socket ( ) .
Fix typo .
Fixed nextstep buglet ; check for libreadline w / o libterm *
closes bpo - 35184 : Fix XML_POOR_ENTROPY option that breaks makesetup parsing of pyexpat line in Setup . ( GH - 13064 )
Strengthen the tests for format ' % Y ' , in relation with issue .
handle class exceptions
Merged revisions 85101 via svnmerge from
merge
Modified to work together with BBPy
Python 3 . 9 . 0a3
Issue : Try the delete demohelp . txt part again .
Have links in OrderedDicts be native Python lists instead
Fix three ( ! ) object leaks in the code for assignment to __bases__ .
Improve the leak fix so that PyTuple_New is only called when needed .
define_module ( ) : Change the " index sub - item " for definitions at module
Outdated .
: Check for errors around the use of PyTokenizer_FindEncoding ( ) .
Merged revisions 70137 via svnmerge from
Untabify .
_PyUnicode_CheckConsistency ( ) ensures that the unicode string ends with a
skip multiprocessing main handling test when Python is built without threads
Issue : Update test_capi
point errors related to nonlocals and globals to the statement declaring them ( closes )
Issue : Merge Readline updates from 3 . 5
Files are now their own iterator . The xreadlines method and module
Add cjkcodecs to build process
Update URLs
Backport 58594 :
Issue : Make webbrowser support Chrome on Mac OS X ( merge 3 . 5 - > 3 . 6 )
Issue : Add diagnostic functions to ssl . SSLContext ( ) .
Issue : os . lchflags ( ) is not always available when os . chflags ( ) is available
Merge with 3 . 3 for Issue .
: fix default value for " buffering " param of open ( ) .
bpo - 31873 : Update unicode . rst - ' unicode ' capitalization ( GH - 4125 )
Remove support for - - without - universal - newlines ( see PEP 11 ) .
Issue : Merge from 3 . 6
bpo - 41123 : Remove PyUnicode_GetMax ( ) ( GH - 21192 )
Merged revisions 78777 , 78787 , 78790 via svnmerge from
Added a note about removing preferences and . pyc files when you move the Python installation .
\ idxcode - > \ py @ idxcode ( mimics index stuff in python . sty - problem only
Added varargs syntax " def f ( a , b , + c ) : . . . "
Added some more documentation to as_string ( ) to make it clear that
bpo - 41325 : Add version note for args and kwargs property in call object ( GH - 21525 )
typo in error message ( fname vs . file ) .
Typo
Update version number to 2 . 5
add support for Python ' s bool type to xmlrpclib - patch # 559288
Issue : Avoid crashes when trying to pickle huge objects or containers
Add renaming notices to 2 . 6 dbm module docs .
fix escape_encode to return the correct consumed size
merge
_structure ( ) : Swap fp and level arguments .
OptionDummy now has a constructor that takes a list of options : each string
Oops , got a little too enthusiastic deleting code in that last revision :
PySocket_getaddrinfo ( ) : fix two refcount bugs , both having to do with
Merge .
Patch : Setting exe_extension for cygwin
Fix Coverity .
use PyErr_SetString instead of PyErr_Format
Improve the test output a bit .
asyncio : Fix from Anthony Baire for CPython issue 19566 ( replaces earlier fix ) .
Accidentally didn ' t commit Misc / NEWS entry on when __unicode__ ( ) was removed
Sync up with trunk . Try to flush repeatedly to ensure the exception is raised .
bpo - 33483 : more correctly handle finding the C compiler ( GH - 6780 )
bpo - 35733 : Make isinstance ( ast . Constant ( boolean ) , ast . Num ) be false . ( GH - 11547 )
: fix typo ( name - > named ) .
added sqlite3 section
[ 755617 ] os module : Need a better description of & quot ; mode & quot ;
add a NEWs entry for r66614
Close : Fix regrtest - - slow with interrupted test
Refrain from using inline suites .
Issue : Create libregrtest / runtest_mp . py
Include the filename in the errror message for F5 without saving .
more complete tests of built - in functions
indicate that mktime is platform - dependent . See SF bug 434143 .
Issue : merge the discordant NEWS entries .
Issue : Various improvements to struct module documentation .
Issue : pickletools . optimize ( ) now aware of the MEMOIZE opcode , can
Issue : Fixed a bug in C implementation of OrderedDict . move_to_end ( )
Close : Inconsistent size for GIL release in hashlib
: Make sure that HTMLParser . feed ( ) returns all the data , even when convert_charrefs is True .
Issue : Correct faulty test file and reference in test_tokenize .
Issue : Fixed the declaration of some public API functions .
Merged revisions 81419 via svnmerge from
new material for 0 . 8 . 1 release
Add - Olimit 1500 to OPT if CC supports it .
* ext . tex : did most of the tedious changes from plain ASCII text to LaTeX .
Fix the sqlite failure - - it was the usual , PyInt_Check - > PyInt_CheckExact .
: add Misc / NEWS entry for e63ac05ccfa8 .
Patch by Tony Lownds , slightly massaged by me : streamline the
more and more implementations now support sys . subversion
Move import lock - related functions to a separate doc section .
Tweaked various comments , docstrings , and error messages .
Coded WHY flags as bitfields ( taking inspiration from tp_flags ) .
Fixed typo , description of changes to dbm module .
Issue 11177 : asyncore ' s create_socket ( ) arguments can now be omitted .
Issue : Match objects in the re module now support index - like objects
Better detection of bad entries in option table .
- Numeric now lives in Lib : site - python .
On ResourceWarning , log traceback where the object was allocated
Issue : Convert _struct module to Argument Clinic
bpo - 37663 : have venv activation scripts all consistently use __VENV_PROMPT__ for prompt customization ( GH - 14941 )
Fix target path in message .
Null merge
Rename History to IdleHistory .
[ Bug ] Add necessary import to example
Skip the test requiring ctypes if ctypes is unavailable .
Forward - port new test from r64300 .
Merge : Be consistent and don ' t use a bullet list when a section has only one item .
Fix instructions on how to rebuild some modules
: merge with 3 . 2 .
Add test for failure of the getattr call in pcre_expand ( ) - - it used
Some more news .
Significant speedup in new - style object creation : in slot_tp_new ( ) ,
Use preferred form of raising exceptions .
Remove unnecessary grammar decorations and change header ( GH - 20819 )
bpo - 39009 : Fix typo in test__locale ( GH - 17544 )
Issue : fix TESTFN_UNENCODABLE for japanese code page
& do_cmd_bcode , & do_cmd_ecode : Removed ; no longer used .
Properly fix SF bug ( Gregor Lingl ) : shellpython2 . 2 - Qnew smart
bpo - 38870 : Add docstring support to ast . unparse ( GH - 17760 )
: add note about checking the exact type in assertIsInstance doc .
Issue : Symbolic group references to open group in re patterns now are
Merge What ' s New fix from 3 . 3
PyUnicode_FromWideChar ( ) and PyUnicode_FromUnicode ( ) raise a ValueError if a
Fix Issue6085 - SimpleHTTPServer address_string to return client ip instead of client hostname
Since Py_TPFLAGS_HAVE_WEAKREFS is set in Py_TPFLAGS_DEFAULT , it does not
Try to fix linking failures under Windows
Issue 8753 : Added documentation for Py_ReprEntr and Py_ReprLeave .
Fix issue13726 : - S accepts an argument just as - - start .
Better variable names
Issue : Added versionchanged for processName .
Speedup and simplify negative counter using count ' s new step argument .
Use universal construct os . path . expanduser ( ' ~ ' ) instead of os . environ [ ' HOME ' ]
Issue : socketserver now restarts the select ( ) call when EINTR is returned .
merge from 3 . 2
bpo - 34625 : Update vendorized expat version to 2 . 2 . 6 . ( GH - 9150 )
Python 2 . 2 as installed by Apple has a version of _Res that is too old . We
Temporary checkin to get configure to report more
Fixed sentence about modulator
Issue : text_textview now passes when re - run in the same process
Adapted unweave to Matthias ' fixes . It ' s still off - by - one - space in some cases .
Update test_sundry . Many modules have now tests , but
Fix sys . flags sequence behavior and add a test case .
Update the filenames in the MSVC project file for Python 2 . 5 .
Issue : Make Control - Mousewhell only change font size and not also scroll .
Merge : : Change generic UCD PropList link to version specific link .
Merged revisions 80240 via svnmerge from
Add < hr > after top navigation bar ; now looks more like the rest of the
Fix typo .
Merged revisions 68462 via svnmerge from
bpo - 40707 : Document that Popen . communicate sets the returncode attribute ( GH - 20283 )
include macdefs . h for declaration of getwd , if non - gusi python .
Fix for SF bug # : can ' t deepcopy recursive new objects
Added a space in an error message
( Modified ) patch by Ping - SF Patch .
Fix frozenset ( ) ref count and a comment typo .
Implement PEP 238 in its ( almost ) full glory .
As discussed on python - dev , add a mechanism to indicate features
Changes to ProgressBar :
[ Bug ] Update version of zlib referenced in text
Suggest how to use compare_networks ( ) with a modern Python .
bpo - 31558 : Update NEWS and ACKS ( )
Issue : Merge 3 . 6 .
bpo - 40874 Update the required libmpdec version for the decimal module ( GH - 21202 )
Closes : use bytes type to send and receive binary data through XMLRPC .
Fix typo in the dataclasses ' s doc ( GH - 8896 )
Address reviews for open ’ s opener argument doc patch ( ) .
: merge with 3 . 3 .
fix loss of a space after indexing markup
Call of _cmp had wrong number of paramereters .
bpo - 34726 : Fix handling of hash - based pycs in zipimport . ( GH - 10327 )
Add a regrtest option to re - run in verbose mode immediately after a test fails , and use that option on the buildbots .
Change markup so it generates a link
Add What ' s New entry for date subclass behavior ( )
Add details about the return value for mmap . flush ( ) .
Merged revisions 75528 via svnmerge from
Added detection for getgroups ( ) , fpathconf ( ) , pathconf ( ) ,
fix for Bug ID 448100 - " test code using NL instead of CRNL "
Issues : remove old cPickle imports from multiprocessing
needforspeed : added rpartition implementation
note that get ( ) is not affected by default_factory ( closes )
Fix XML vulnerability link targets .
As part of fixing bug , add a test case for string . zfill ( ) with Unicode
bpo - 41215 : Don ' t use NULL by default in the PEG parser keyword list ( GH - 21355 )
Doc : Missing backquotes . ( GH - 11982 )
merge with 3 . 4
Some more tuning of quicksort : use pointers instead of indexing .
asyncio doc : explain why the loop is running twice
Issue : Port test . test_importlib . import_ tests to use PEP 451
Temporary hacks to arrange that the pickle tests relying on protocol 2
fix typo ( )
Add NEWS for OpenSSL changes .
Added about dialog
Closes : avoid shadowing builtin input ( ) in example code .
bpo - 32598 : Use autoconf to detect usable OpenSSL ( )
Post - release fixups for Python 3 . 5 . 1 .
Fix mixed mallocs : re - > re_patbuf . buffer is allocated with std malloc ( ) .
Issue : Ported from 2 . 7 tests for non - ASCII data .
Issue : Optimized io . BytesIO to make less allocations and copyings .
Issue : Backout fix for because of use of unsafe subprocess . call ( shell = True )
If you created a weakref in an object ' s __del__ method to itself it would
Merged revisions 75524 via svnmerge from
Patch : Update RiscOS support . In particular , correct
install svn in / usr / local / bin and remove from ~ / local / bin
Remove " ast " aliases from the parser module .
bpo - 36763 : Add _PyInitError functions ( GH - 13395 )
Mini - PEP : Simplifying numbers . py
: Handle readline . __doc__ being None in site . py readline activation .
Fixed : distutils . cygwinccompiler . get_versions was trying to use a re string pattern on a bytes
Markup fix
Move printing of filename and lineno to tb_displayline .
support linking the _bsddb extension module against BerkeleyDB 4 . 5
Minor edits .
Begin writing tests for importlib . machinery . PathFinder .
Issue : Improved cross - references in the xml package .
Issue : fix typo
Rebuild the wininst . exe files .
SF - modulefinder fails to find extension modules in packages
don ' t identify the toplevel namespace by name
Backport r82456 .
Merged revisions 73145 via svnmerge from
Dedent a paragraph that was accidentally aligned with a preceding
bpo - 38862 : IDLE Strip Trailing Whitespace fixes end newlines ( GH - 17366 )
Fix zipimport . c ' s read_directory ( ) to use appropriate types for the values
Issue : Fix potential reference leaks in bytearray . extend ( ) .
Issue : Clarify the return value of the scheme attribute of ParseResult and SplitResult objects .
Fix a very minor ( but annoying when looking for things ! ) markup nit .
PEP 308 implementation , including minor refdocs and some testcases . It
Merge : Fix the return value of BZ2File . _read_block ( ) to be consistent with comments .
whitespace
Issue : io . FileIO ( ) . write ( ) clamps the data length to 32 , 767 bytes on
merge with 3 . 4
Merge
bpo - 42749 : Fix testing bignum if Tkinter is compiled with Tk 8 . 4 and dynamic linked with Tk > = 8 . 5 ( GH - 23955 )
Typo fix - " mesasge " should be " message " ( GH - 22498 )
merge
Doc : add missing capture_output arg to subprocess . run ( ) signature ( )
Merge with 3 . 3
Use new name for GetoptError , and pass it two arguments
Add news about csv module updates .
added warnings about security risk of using tmpnam and tempnam
Issue : Drop _XOPEN_SOURCE on Solaris .
Closes : returned numeric value for ' FATAL ' logging level .
Simplified inclusions and avoid prototypes copied in from elsewhere .
bpo - 31642 : Restore blocking " from " import by setting None in sys . modules . ( )
Merge 58450 : fix uncollectable reference caused by bsddb . db . DBShelf . append
Branch merge
Issue : Switch urllib . parse to use RFC 3986 semantics for the resolution of relative URLs , rather than RFCs 1808 and 2396 .
Merged revisions 76978 via svnmerge from
bpo - 31836 : Test_code_module now passes with sys . ps1 , ps2 set ( )
Added hint to skip the heavy stuff on first reading .
Added provision for THINK C which doesn ' t have sys_nerrors or sys_errlist .
This commit was manufactured by cvs2svn to create tag ' r23c2 ' .
: merge with 3 . 3 .
Merge
Merged revisions 86192 via svnmerge from
merge 3 . 3 ( )
Add get_objects function . This is a low level function ( like
Stop Python code from instantiating a new stdprinter with sys . stderr . __class__ ( )
bpo - 32436 : Add docs for contextvars ( )
Issue : Fix test_pdb on Windows
Fixed to understand new file format ; and cosmetics .
Issue : Remove dead code
Add 3 . 0b3 uuid
Issue : Silence resource warnings in test_socket . Initial patch by Xiang Zhang , thanks
correct description of list . insert ( )
Merged revisions 78784 via svnmerge from
missing word ( should backport - is release23 - maint still locked ? )
Merge with 3 . 3
bpo - 42361 : Use Tcl / Tk 8 . 6 . 10 when building the installer on recent macOS ( GH - 23293 )
+ Text - mode ( but not yet GUI mode ) pydoc now produces useful stuff for
Document the new , netrc awareness in nntplib .
Remove unused variable
Issue : Handle better a race condition between the interactive interpreter and
Revert " bpo - 38691 Added a switch to ignore PYTHONCASEOK when - E or - I flags passed ( ) " ( GH - 18553 )
New style section headers .
Fixes for Windows ( but also tested on Linux ) . Test suite now completes , and this module should not leak in the face of errors .
Remove a typo . ( Or is this some markup I don ' t understand ? ; - )
Issue : Fixed possible buffer over - read in bytearray . The bytearray
Update PR template file , don ' t reference bpo 12345 ( GH - 5897 )
Fix whitespace .
Fix webbrowser . BackgroundBrowser on Windows .
Closes : removal of two unused variable in locale . py
Fix Issue10012 - httplib headers , which are ( sometimes mistakenly ) int are explicitly cast to str ( bytes - in py3k ) .
Raymond - Hettingers - MacBook - Pro : py27 raymondhettinger $ cat svn - commit . tmp
New version handles - o < string > same as - o < string >
Tim ' s version 4 , with my mods
Backport source links from 3 . x .
merge with 3 . 2
: mention " deleting " as an alias for removing files .
Centralize the processing logic for LaTeX documents into the mkhowto script ;
bpo - 33789 , test_asyncio : Hide PendingDeprecationWarning ( GH - 7461 )
bpo - 34987 : Fix a possible null pointer dereference in _pickle . c ' s save_reduce ( ) . ( GH - 9886 )
Issue : POSIX semaphores are broken on AIX : don ' t use them .
Gave intersection_update a speed boost .
bpo - 32565 : Add missed versionadded directives for all new opcodes . ( )
Closes : Fix unnecessary recompilation of OpenSSL on Windows
Hint for C + + users on AIX by Gary Duzan added .
Document that atexit execution order is undefined ( )
Use re instead of regex .
Merge issue 1294232 patch from 3 . 2
bpo - 31392 : Update macOS installer to use OpenSSL 1 . 0 . 2m ( )
Issue : Document that py_compile now uses importlib for its file
Merged revisions 67028 , 67040 , 67044 , 67046 , 67052 , 67065 , 67070 , 67077 , 67082 via svnmerge from
Issue : BTPROTO_SCO supports only bytes objects
: small correction to atexit docs .
Added names to all widgets
Fix a compile warning in dictobject . c ( GH - 16610 )
Make string . translate ( s , table ) work for Unicode s . Two things are
bpo - 42635 : Mark JUMP_ABSOLUTE at end of ' for ' loop as artificial to avoid spurious line events . ( GH - 23761 )
Change [ _Py_ ] re_compile_pattern ( ) to return a char * .
Merge heads
Removed extra period from \ versionchanged entry ; macro adds period
Remove unused ' search_dir ( ) ' method .
Issue : Fixed memory leak in TextIOWrapper . tell ( ) in rare circumstances .
: add validate option to base64 . b64decode
Update release date .
PyRun_InteractiveOne ( ) ,
Patch [ 972332 ] urllib2 FTPHandler bugs / John J . Lee
: add Programming FAQ entry about the behavior of closures .
Make socket . getservbyname test optional on socket module having that
Issue : File open dialog now works instead of crashing
Add symbols of the stable ABI to python3dll . c ( GH - 23598 )
Decode untyped strings from UTF - 8 .
Replace backquotes with repr ( ) , to silence a SyntaxWarning .
( Merge 3 . 2 ) Issue : Fix FileIO . readall ( ) on Windows for files larger than 2 GB .
added index entries for __ * __ identifiers
Clean up unused imports for the peg generator module ( GH - 19891 )
forward port of 1 . 136 . 6 . 2 :
Issue : Get rid of WindowsError . Use OSError instead
This file ' s previous log message is bogus .
Remove extraneous " contains " - - pointed out by Daniel Barclay
Add note that difflib was added in Python 2 . 1 .
: remove sgmlop support from xmlrpclib ; the sgmlop parser does not do much validation and is no longer much faster than e . g . the cElementTree XMLParser .
[ merge from 3 . 5 ] - Prevent HTTPoxy attack ( CVE - 2016 - 1000110 )
Add more tests for hashlib and hash object attributes
Merge
: add newlines = ' ' to csv . writer docs .
Advertise nesting directives for class / method and class / data combos .
: remove redundant PyLong_Check calls when PyInt_Check was already called .
Obscure marshal fixes :
SF patch ' the the ' typo . Contributed by George Yoshida
null merge
* minor tweaks relating to the package nature of the beast
_PyStack_UnpackDict ( ) now returns - 1 on error
minor correction . python 2 . 3 was released with the bsddb module 4 . 1 . 6 not 4 . 1 . 1
Correct long standing bugs in the methods for random distributions .
Issue 27744 : AES - CBC and DRBG need Kernel 3 . 19 +
Issue : merge from 3 . 2
added setentry ( ) and use it in commit so CVS / Entries gets updated
Merged revisions 84239 via svnmerge from
require C99 bool
Match current tracker name , though I do not know if still active .
removed unused globals
Fix indendation level of versionchanged directive in configparser . items ( ) doc ( GH - 7543 )
merge 3 . 4 ( )
Issue : Document ctypes array and pointer classes
Document the subprocess Popen . args attribute ( issue21353 )
- The repr ( ) of a weakref object now shows the __name__ attribute of
Merge heads
Copy the wsgiref package during make install .
Add corresponding support for the alltt environment to the HTML generator .
Issue : Fix typos in Lib / .
Use the extended library search path when looking for readline ( simple
Add " suspicious " builder which finds leftover markup in the HTML files .
curses HOWTO : fix some PEP8 and a code example to actually raise the exception it promises .
Added ( limited ) compuserve error parsing
- The mac - specific tests should also be run on darwin .
magic number changes of CR / LF are swapped
Call super in Generic . __init_subclass__ ( )
Merge 3 . 6 ( issue )
Issue : Fix a stack corruption in ossaudiodev module when the FD is
Adding TextTestResult to unittest . __all__
bpo - 29737 : Optimize concatenating with empty tuple . ( )
The turtle docs will be moving to another location in the Tkinter chapter .
bpo - 34485 : Enhance init_sys_streams ( ) ( GH - 8978 )
Patch : Expose compilation errors as exceptions on request .
Merge typing docs cleanup diff by Zach Ware from default back into 350 branch .
Merge 3 . 5 ( NEWS )
Conform to standard boilerplate .
Removing this directory ; it ' s no longer needed now that Misc / Makefile . pre . in
Merged revisions 79674 via svnmerge from
Issue : Clean up C - API docs for PyLong_AsLong ( AndOverflow ) ; clarify that __int__ will be called for non - PyLongs
group ids and user ids can be longs now ( )
classify abs ( ) argument type ( GH - 8103 )
Fix uses of the default role .
Support the use of bibtex . This is transparent to the user .
Fix problem handling EXTENDED_ARGs from SF bug # 1333982
Issue : Fix a missing import in the unittests .
Issue : Merge from 3 . 6
For readable ( ) objects , the previous value of ' flags ' was ignored .
Recorded merge of revisions 72708 via svnmerge from
Remove spurious space before a period .
bpo - 37354 : Sign Activate . ps1 for release ( GH - 15235 )
Rich comparisons fall - out :
Reformat _tkinter code to follow PEP7
M - x untabify
Merge 3 . 6
Relocating file to Lib / lib - old .
Update idlelib / NEWS . txt
A different binhexer , apparently : - (
Remove old reminder
Added note about need for - traditional - cpp on the MacOS X Public Beta .
I ' ve created three bat files based on the howto http : / / wiki . python . org / moin / Building_Python_with_the_free_MS_C_Toolkit
Convert two other post - decrement while - loops to pre - decrements for consistency
Merge from 3 . 3
Merge from 3 . 2 ( , , )
Call SchedParams ( 1 , 0 ) so the dialog is correctly re - enabled after something has been on top of it .
Issue : Check warnings messages in the aifc module tests .
Merged revisions 77589 via svnmerge from
bpo - 41336 : Fix the error handling in zoneinfo_new_instance ( ) ( GH - 21546 )
Revert revisions 57472 , 57474 , and 57477 which disabled some tests
Got rid of an extraneous semicolon .
Consistency : In two places , change { \ it . . . } to \ emph { . . . } .
Rearrange include headers ( needed on some platforms ) .
Make the original , makesetup - based , targets for building shared modules
starting distutils . ccompiler test coverage and cleanup
Issue : Moved common code for the replace ( ) method of bytes and bytearray
Issue : ssl . cert_time_to_seconds ( ) now interprets the given time string in the UTC timezone ( as specified in RFC 5280 ) , not the local timezone .
tracemalloc : Fix slicing traces and fix slicing a traceback .
bpo - 37007 : Implement socket . if_nametoindex ( ) , if_indextoname ( ) and if_nameindex ( ) on Windows ( GH - 13522 )
Close issue20653 : allow Enum subclasses to override __reduce_ex__
Patch by Charles Waldman to implement an optional nlines argument to
Added support for 24 - bit rgb movies
Add fast - path in PyUnicode_DecodeCharmap ( ) for pure 8 bit encodings :
