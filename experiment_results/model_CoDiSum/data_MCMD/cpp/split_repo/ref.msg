remove unnecessary and incorrect std move which makes some compilers unhappy
fix rpc return type
fix a bug in vehicle s position settings
add exception for mocap message
add include for stat h
dronecontrolbase cpp solve typos
assigning fpv_pawn when there are pawns in environment
x platform path for examples
remove unneeded includes
fix droneshell return type is string not bool
fix rpc return type
fix killrecording when recording is never started
add call to updaterenderedstate for simmodecar
merge pull request from aburgm macos exe path
use recvfrom for reading data from socket in macos fixes microsoft airsim
revert disable cameras after fetching images projection matrix
reduce thread priority
merge pull request from mit acl ros catch empty img
linux joystick isconnected fix
add heartbeat for sitl mode
minor local var name change
revert change for compatibility
temporarily restore preprocessor checks for c std
fix mavlinkdronecontrol explicit call to constructor
remove namemode for actors to be spawned for bwd compatibility
fix osx compile error
fix build on macos platform
better fix for takeoff bug
add ffe strength validation
merge pull request from wardzacharyj fix sim collision info
main cpp solve typos
combaseapi h incompatibility fix
fix error type of first parameter of simgetimage
add typedef for uint
safetyeval cpp solve typos
solve strict overflow
vehiclepawnbase cpp solve typos
fix unit test
reduce random test iterations to speedup travis
suppress float equal in readertest cpp
remove debug code
fix a clang gcc warning
add a test for biginteger operator
add simple test for unchanged dom after parse error
fix compilation in schema test
further improve coverage of genericpointer erase
add key string forwarding to the capitalize example
valuetest add test for uint64 comparisons
add a must underflow case as suggested by pah
add simple test for issue
fix a istreamwrapper test with incorrect type
workaround of sortkeys example
improve value deep clone coverage
improve coverage for schemavalidator appendtoken
with recent clang when expected is false this code triggers wunreachable code
fix windows build
add jsonxwriter rawnumber
fix clang warning
improve utf8 encode coverage via writing to autoutf
disable patternproperties tests when no regex
remove unncessary std move
add capitalizefilter rawnumber
fix yet another wfloat equal warning
workaround for valgrind false alarm on wcscmp
cover memorypoolallocator capacity
don t use nullptr
add missing flush in examples flush will be called in writer later
fix a bug in perftest was asserted in debug
improve reading kparseerrortermination coverage
disable failing test for now
add reproduction test case
add unittest for state transition to iterativeparsingmemberkeystate
remove superfluous typename
disable parsebyparts example for clang
disable including codecvt in tests as many libraries does not support it yet
add unicode regex test
add a unit test for skipwhitespace
try to fix a compilation error on gcc clang
fix gcc warning
try turning on slow test on number parsing
try to fix effc warning
update readertest cpp
update test unittest pointertest cpp
fix schema test compile error
further improve skipwhitespace coverage
improve coverage of stack
update vnxtpmd cpp
update vnxtptd cpp
mod support custom resume type for sopt gateway
update vnxtpmd cpp
update vntapmd cpp
update vntaptd cpp
update vntaptd cpp
update vntapmd cpp
fix issue in tests
update example cpp
remove rotation from boost bench
update utils cpp
moved pattern formatter from spdlog details to spdlog
update bench cpp
fix unused warning message
small fix in comment
fix multisink example
update example cpp
solve compiler warning in tests
fix macro tests
remove unused template from spdlog cpp
remove empty line
fix typo in example
wip static lib
fix example cpp
add drop_all to example
minor fix to test
another test of stupid script
merge pull request from danielchabrowski fix registry test
update utils cpp
fix bench tidy warning
cygwin bench fix using atoi
update example cpp
fix windows setenv check
add the all flag to bench formatter
fix multiple clang tidy warnings
moved pattern formatter from spdlog details to spdlog
fix a test case error
add std to vector decl in bench
revert last commit
fix warning in bench
fixd copy ellision warning in tests
updated bench cpp
update tests utils
merge pull request from podsvirov increased tolerance waiting time for mingw
fix valgrind warning in example
update spdlog cpp
update spdlog cpp
remove warning from example cpp
minor fix in example
updated formatter bench
updated formatter bench to include source location
error if compiling spdlog cpp under header only configuration
improve circularbuffer error message
fix intermittent invalidimagelayout error for main depth buffer
update view cpp
vulkan fix mipmap generation on android
fix regression with getdefaultinstance
more accurately evaluate the size of msaa textures
vulkan fix regression with window resizing
always wake up a job queue when a job is ran
swizzle textures when swizzle is enabled not the other way around
move call to endframe after commit
fix incomplete fbo when rendering to cubemap
fix bad size in ktx reader
remove unusued function
meshassimp fix gltf material hash function
reduce watermark console spew
honor clear color fixes
disable rounding fbo sizes on webgl
fix a jobsystem bug in waitandrelease
fix vulkan swapchain size on windows
use the correct size for the imgui glyph atlas
enable custom allocator test for windows
fix jni binding
fix ktx image test on windows
opengl driver allow point sprites on desktop
properly initialize vulkanbinder mdescriptorkey
fix msan use of uninitialized value in destroyvertexbuffer
fix can t set material parameters on default material instance
fix windows build
fix framegraph test
fix struct initialization
gltfio do not loop animation samplers fixes
apply clipspace fixup in glslpostprocessor
frameuniforms resolution gets consistent precision between vs fs
fix minor vulkan warning
fix light vector transform
fix incorrect camera view matrix with non rigid transformations
fix warning about macro expansion
fix unit test failure
fix cmgen test
use std move in meshassimp
vulkan warn instead of panic for sampler overflow
fix linux android debug builds
vulkan disable unreliable blit diagnostics
fix the surfacecapabilities check
remember the high intensity for directional lights
fix refracted objects drawing order
vulkan fix destroy while used for ubos
vulkan log driver version info
fixes linearimage memory leak
do not clobber vk_icd_filenames
remove an incorrect assert in shadow mapping
remove unnecessary using
vulkan request the driver info extension
pass shared context on windows to wglcreatecontextattribs
minor fixup to suzanne vulkan demo
improve depth test usage and tracking
fix const warning
vulkan fix zealous frame skipping on android
fix regression with non vulkan backends
jni fix for badly sized pixelbufferdescriptors
js fix use before free for png data
repair web samples by fix typo in previous fix
fix mutex lifetime in waitforcommands
shadowmap fix potentially wrong entitymanager
disable hex mode after printing diagnostics
fix vulkan black screen on windows with nvidia hardware
improve binarytreearray unit test
assetpipeline renormalize after baking transforms
remove usage of the invariant keyword
fix debug builds
terminate tonemapping material
minor cleanup of postprocessmanager ctor
suzanne fixup create entities for point lights
make we our indices to the hammersley sequence are within range
postprocessmanager another vulkan related re ordering
fix mipgen windows build
fix cubemap mirroring
repair vulkan addblob should return an index
turns out the noopdriver is needed in some tests
fix flaky sstream test
better fix for texture unit spew
gltfio fix skinning offset
fix stale ibl string in shadowtest sample
fix crash when using free flight and hovering outside of the window
fix crash in strobecolor test
repair vulkan sychronization fixes
add quiet flag
use uniform sampling for cloth dfg
fix bad skybox color in web demo
fix typo in ibl computation comment
fix recent cmgen ktx regression
vulkan add missing field to vmavulkanfunctions
only log no texture bound in debug builds
uhm actually set the ibl rotation from the builder
fix bad memory access in js wood demo
update mainwindow cpp
trivial typo fix
fix plotting with datetimes on x axis
fix extension loading
cvsparse used wrong var for last row check
cipher fix leak and database being locked when opening a normal db file
replace misleading text
mainwindow don t cancel after one insert or the like in sql tab
fix opening of edit dialog when pressing the return key
do not reload not needed information when changing the current tab
fix crash when trying to open non existing table via command line
don t hijack the enter key in the browse data tab
also fix rowid column being shown when changing the sort order
minor wording tweak
fix crash when clicking create index in empty database
fix usage of emit macro
attempt to fix the build on some platforms again
put full cell data into insert statements when dragging tables
remove useless debug message
fix typo in comment
add ctrl l shortcut for log view
speed up csv import by not querying the stream position
dbhub show a warning when trying to push unsaved changes
data browser avoid showing blobs containing unprintable characters as text
plot fix crash with selecting column out of range
use edit role when querying values for completer in browse data tab
sqlitetablemodel don t insert new rows when there is no new data
edittabledialg table without rowid don t have to be integer pk columns
change format of hex display format
set adjustment rules for combobox within foreign key editor
preferencesdialog correctly send events on the frames to the parent
formatting tweak to assist code clarity
adapt to qscintilla behaviour with windows line terminator
avoid style sheet inheritance in colour chooser boxes in preferences
edittabledialog also update the sql when type is automatically changed
also emit loading finished signal when fetching an empty table set
tests fix tests
remove unused code
fix incorrect display of sql calltips containing non us ascii characters
update testregex cpp to fix travis build
plot remove axis labels when there is no plot
adds a missing space between sentences
reset sorting when changing the current table
enabled line breaks in schema removal by default
revert remove useless debug message
fix correct row insert
also set foreign key preferences when creating a new database
fix binary in place editable cell
merge pull request from kehugter master
issue adjust sql schema column in db schema dock to content
remove unnecessary include
right click on vertical header to delete the selected record s
sqlitemodel fix endless loop if offset is used in limit
revert test code
add x rows affected part to the execute query results
sqlitetablemodel don t crash when query contains
remove deleted files from recently opened files list
set default color to black
always set sort direction to ascending when sorting a new column
mac version references incorrect shortcut in fk navigation
reload schema when refreshing the browse data view
dbhub fix progress dialog when fetching pushing db after cancelling
fix memory leak when attaching a database
fix crash and blocking condition while remove hidden columns at copy
fix crash when opening a second database
grammar don t drop constraint name for one field unique constraints
mac version references incorrect shortcut in fk navigation ammend
clear edit dock after closing database
update recordset label when changing the filters
fix copy in internal buffer
try fix deadlock
sqlitetablemodel fix wrong limit value in fetchdata
use the column type affinity for the default text alignment
fix build for older versions of qt
merge pull request from achpile master
edittabledlg missed to remove another textchangeevent
editdialog don t add file filters on os x
comment out additional gotreply signal connection
fix warning when browsing views
make result table in sql tab non editable
valid html clipboard content
double click a column for selecting it
remove unused variable
fix my stupid log appending
support filtering for null values
crash when running query with columns selected for plot
tests fix crash in regex tests
no need to search views for foreign keys
exportsql only quote if text or blob datatype
avoid warning when creating new databases
preserve format in copy create statement
fix build error
createindexdialog sort tables names alphabetically
editdialog fix text editor when null chars are typed in hex editor
remove leftover debug code
edittabledialog make it possible again to delete default value
dbhub don t tolerate host name mismatch errors
plot wrong date format
improve error messages when importing sql files
also get default values after add row to table with keyword as name
merge pull request from bernardosulzbach bad sentence fix
plot no selection for tree plot columns widget
a new approach to make the editdialog appear centered over the main window
make sure to register the regexp function when creating a new database
don t create savepoint when releasing all savepoints
always load settings for browse data tabs
exportsql incorrect extra after create table
fix infinite loop when executing some sql statements with limit part
sqlitetablemodel return from setdata if value hasn t changed
add keyboard shortcuts for recently opened files actions
and another type mismatch on mac
save state and geometry of the main window
grammar mark named table constraints as unimplemented
auto completion for column names in the sql editor
grammar jump over column definitions after parsing them
add missing braces
importcsv fix compile on clang
fix possible crash when loading project files
also remember databases with no objects
don t ask for pk again when chaging back to a editing enabled view
fix endless loop in edittabledialog addfield
set override cursor if selecting a line
fix crash when searching in an empty table
qhexedit fix initialization order in qhexeditprivate
update schema after attaching detaching a database in an execute sql tab
fix editable browse data table items
silence some warning
use a more sophisticated method for detecting image data in edit dialog
fix regexp implementation to work with unicode characters
add unittest for foreign key parser problem
remove syntax highlighting for tables when db is closed
remove an un needed apostrophe from a user visible string
plot suggest file formats in save dialog
grammar fix parsing virtual tables
editdialog fix window positioning
plot load all data before drawing a plot
class nulllineedit lacks q_object macro
cipher fix memory leak when cancelling the cipher dialog
databrowser let the view update the model when sorting
fix line numbers position recalculation
untranslatable message in sql log
updater show the url we got from the release file
remove the default property of the fetching dialog
clear browse data paging label on database close
don t enable the attach database button if there is no db open
optimise success message in execute sql tab
merge pull request from revolter hotfix rowid refresh
mainwindow update execute query table view when no results returned
fix issue with inserting rows into single column pk tables
merge pull request from schdub master
improve error handling when saving
show temporary tables and views in structure and browse tabs
fix broken table object sqltext edit update
dbhub improve file size format
fix text detection for blobs containing bytes in ascii range up to a zero
add absolutepath to the recent menu fix
scale large images in imageviewer to fit the viewport by default
explicitly specify the plugins directory on windows systems
refresh table views when using table command line option
change wording of a message box
fix editing of collation in new fiews in the edit table dialog
replace obsolete key combination in comments
do not prompt for saving project file on close when it is just open
add setting to allow user to change the background color of null value fields
add possibly missing include
compiler warning fix by bokic users sourceforge net
dbhub do no show error message when download was cancelled
syntaxhighlighter fix quotation matching
fix build on windows
dbbrowserdb fix bug in updaterecord
don t canfetchmore on our own
show tooltips in db structure views
dbschema shortcut for schema pane
activate one more test
refuse from previous commit
don t treat numbers as json data in edit cell dialog
add selecting fields in execute sql tab for edit cell dock via keyboard
merge pull request from mkleusberg master
fix a crash with browsing virtual tables that we can t query
qscintilla finetune auto completion
mainwindow add extra shortcut for browse data refresh button
don t forget the current selected table when switching tabs
after closing an sql tab set focus to the new currently selected tab
pluralised pixels label in edit cell
use platform specific path separators in window caption
store a relative filename if the path to project and to db is the same
fix not working save button in the message box when closing a database
extendedtablewidget fix bug when just copying one cell
use begin end resetmodel
also load sqlite extensions when importing a sql file into a new db
windowsupdatecheck fix a small memory leak
execsql use the correct statement length
fix additional space if we have no completionprefix
don t refresh browse data tab if it isn t visible anyway
fix memory leak coverity
fix importing csv files with no trailing line break and only one column
remove magic numbers
dbhub add some line breaks to the input dialog text for cloning
show status message when saving to project file
revert don t canfetchmore on our own
fix build on some platforms
fix locking the database by just opening it
dbhub clear branches combo box when closing database
remove unnecessary include
only update contents of the edit cell dock when its visible
add a comment about a problem we should fix one day
also enable extension loading when creating a new database
don t show foreign key tooltip within execute sql tab
allow table names consisting only of spaces in edit table dialog
editdialog clear after closing database
fix copy paste edit error in sqlexecutionarea savestate
when searching for a column by name make the search case insensitive
avoid setting the db dirty with some trailing whitespace and comments
change company name
remove some todo notes
revert try fix the loading of the translations of the qt strings
fix message field already exists for non existent fields
highlight column headers to emulate spreadsheet behaviour
set horizontal scrollmode of result widgets to per pixel
send the fetched signal in rowloader even when no rows where fetched
dbhub fix possible crash
remove null in margin of editor in edit dialog for empty cells
updater fix a crash and handle redirects
sql export uses wrong line terminators for windows
use qfontinfo so windows chooses a correct font
fix foreign key preview for foreign keys in the first column
show consistent slashes in recent files menu actions
restore status bar calculations in browse data tab
replace foreach macro by range based for loop
forgot a after var rename
don t paste null value when pasting from empty clipboard
fix loading settings in export sql dialog
add missing include
change the newline character for csv export depending upon the platform
don t add special tokens to vocabulary
revert pr remove unrelated code
merge pull request from lissyx fix invalid prefix
don t oov_score on empty prefix
merge pull request from mrstegeman alphabet logic
load kenlm with lazy
address review comments
lipstick fix indentation
improved code coverage
fix for msvc
two more test cases for numbers
another test case
fix test case in pr
add unit test for parsing deeply nested array
disabling strtod pre check since can t get locale specific behavior to manifest in appveyor
add back forward declaration to address warnings
improve test coverage
disable reserve tests when exceptions are disabled
attempt to satisfy coveralls by add a test for unchanged operator std string
rewind is still not fix
hid an expensive test case
add test cases
overflows are handled wrong by valgrind
hammer fix fuzz code to avoid false positives in case of discarded values
add simple test for cbor byte
checking erasable concept
ambulance fix for
fix a gui bug of folder as workspace
add a philosophy quote in easter eggs
add sci_lineendextend to parameters cpp
make open explorer with expeacted path in folder as workspace
fix typo in code
fix poor performance of hex xml entities
merge pull request from cmeriaux gotonextindicator
add ctrl r shortcut for reload from disk command
fix for monitoring doesn t detect changes
rename sortalllines sortentiredocument
add esc key in function list to switch to edit window
correct build issue on donho master
disable couple of file menu items when it s needed i
fix functionlist not working for javascript regression
add toggle for character panel history clipboard in shortcut mapper
fix maintain indent not working in javascript regression
fix encoding radio button issue in new document
fix crash issue by checking unchecking disable extension column in preferences dialog
fix loading localization file l not working regression
scroll visible area horizontally so a findnext match is fully viewable
allow remove consecutive duplicate lines command to be mapped via shortcut mapper
don t throw pointers mfc habit
make view in browser commands macro recordable
avoid unnecessary notifications
simplify and fix memory leak in getspecialfolderlocation
avoid build issue due to difference between declaration and definition int_ptr vs bool
fix view not closing of not active
tab context menu bug fixes
remove a warning of compiling
fix file rename failing when new name is on a different drive
fix open file dialog error for a long path
new allow users to disable easter eggs in notepad by add noeastereggs xml in notepad installed directory
fix save function in windows dialog
correct whitespace merge difference to master
fix indent indicators continue to following code blocks for python
fix url is not clickable in certain stuation while wrapped is enabled
fix memory leak
fix crash on styler dialog of user defined language dialog
fix hanging problem while shutingdown and backup file is absent
add l argument for hongkongcantonese xml
make end of line conversion operations macroable
revert revert fix multi line tab button stay pushed issue while swiching off
fix save all button is still enabled problem while no file to save
go to correct position after word completion
fix function list focus problem on double clicking
fix find in files tab translation issue on initial dialog call
display file name as root of workspace instead of workspace
improved isalldigits algo
retain field values when disabling status auto detection
fix drag and drop tab regression
expand environment variables in backup path
typo fix a typo
add some languages for l flag of command line
fix an issue in window dialog
fix backup not fully disabled issue
fix preventing find all curr doc working in macro issue
fix bug with smart highlight of data and end of file
fixes identical sub expressions for certificate verification
fix notepad blocked in notification zone issue
add new feature double click splitter resets panes to equal size
fix fix the compiling problem for ansi build
fix the regression of jump line failure via command line n
fix the file status auto detection setting not being set correctly
when sorting reserve enough space for work output
fix the restoring problem from system tray when running in admin mode closes
fix a crash issue in plugin admin
disable the unused gripper of toolbar fixes closes
eliminate garbage displaying in find result fold margin
fix issue with wrong smart highlighting when it is disabled
safely convert eols
nosession must not overwrite config xml fix
autompletion enhancement remove unwanted symbols
fix close all unchanged for sub view
fix the focus not on the opening new file issue
ui only set font quality if cleartype is enabled fixes closes closes
fix a localization regression
fix replace in files regression
update minor change
fix urls not underlined immediately after copy paste
fix block selection cursor position after tab
exception treatment throw object instead of pointer
euro fossa fix a vunerability from search on internet command
fix style not enabled if restyle requested on unstyled document
fix url highlight lockup
correctly identify when a macro is playing
optimize font quality for lcd
fix check sign in encoding menu not shown issue
fix bug of monitoring not working for files under root
remove excess margin with multi line
fix crash while right clicking on docswitcher s column bar
fix crash issue while moving find in files progress window
remove treeview beep when pressing return key
remove the todo to prevent from the useless modification
prevent unfolding from hiding goto line
fix slow tab switching while folding restoring
new author michael allow drop files onto doc switcher and other side panels
fix a typo
fix single quoted string being badly recongnized as attribute value in xml
fix the long time bug that non exist folder to pass via command line is not opened without warning
update docmap when setting user or external lexer
fix the opened zero length file not saving bug
prevent from setting updater s proxyas notepad is not in admin mode
fix tab close button remain pushed issue
fix a bug where the document map highlights incorrectly when the view is scrolled past the end of the file
disable integrity checks in debug
fix wordcompletion bug api file had to be present to function
prevent from crash in tab settings
fix plurals in find replace dialog
fix url detecting issue
fix scroll to last line problem after main window resize
fix auto updater setting toggles between startups bug
fix a regression regarding notepad replacement issue
fix regression of auto indent
fix a crash issue while doing a column insertion on a cjk unicode document
fix file not dirty while its content changed on the hard drive
fix extra space between untitled_str and document number when creating a new document through the function filemanager newemptydocument
fix saved duplicated file in the same view issue
fix accented characters not found in find in files issues
set indicators to draw under the text
fix line ending changes are not detected in reloaded changed files issue
skip auto complete of xml processing instructions
fix auto insert for xml comment incorrect insertion closes fixes
fix incremental search bar option bug
merge pull request from milipili master
add missing macro commands fixes closes
update split lines uses the edgemode if enabled
fix std uncaught_exception deprecation warning in scopedmessage
c header updates
fix build by annotating the new test as shouldfail
cleanup nextafter workaround
add missing cstring include to stringref cpp
hold translated exception in local string while matching
finish fix invalid compilation using mingw
explicit ctor used to fix compilation failures caused by copy
change how non nullness is enforce in stringref constructor
fix invalid isspace call detected by prefast
fix os x compilation error
avoid recalculating string literal size on root tracker construction
fix missing include and wrong comment format
xmlreporter add information about rng seed
check for empty expression properly
suppress warning caused by ntstatus h inclusion
add automake and tap reporters to selftest s main
test case for infinite loop
remove signal handler test for now
only start section s timer if the duration will be used
silence gcc s wunused function firing in compilation tests
initialise unexpectedexceptions in junitreporter
do not open the output file twice
fix segfault during construction of session
don t use console colour if running in xcode
use stringref literal for unknown enum values instead of string lit
use the right overload of std nextafter in tests
fix typo in benchmark name column width calculation closes
remove spurious test following merge
hexescapechar restore stream
explicitly inline trivial function in cumulativereporterbase impl
move stringref s impl details to anonymous namespace
replace ostringstream with reusablestringstream in consolereporter
do not blindly access empty stringrefs
qualified path to xmlwriter to fix travis error
suppress strncpy copy warnings on msvc
remove useless test
add android implementation of writetodebugconsole that prints to logcat
make experimental capture work on windows with read write temp file behavior
disable unreachable code warning for trickytests cpp
unref udp if exists
switch ssdp send arguments around
not keep freed pointer of uart handler
corrected stack start and end in stack_thunk_dump_stack
wificlientsecure rx overflow fixes
fix _useclientmode _useapmode in sdk auto connect mode
wificlientsecure display certificates when debugging is enabled
small fix to stop crash on unrecognized packets
printf to print instead of write
make random more random
allow disabling global serial and serial1 object
update parsing cpp
use polledtimeout for busy loop timeout
merge pull request from gtalusan master
add missing includes to mdns library
schedule_recurrent_function_us should be in iram
libraries fix no ok ack for signed update
report connected when any bytes are available
fix huge leak
adjusted buffer size algorithm
some speed optimizations
clear mdns queryservice s previous result when called
infinite loop while passing file fs h resolved
write loop could lead to wdt with low bitrates or large buffers fixes
http client allow slow streams for sendrequest
make dnsserver debug output like other esp libs
fix slashes in spi library
merge pull request from b pass client fix
wificlientsecure match cn and sans ignoring case
fix basic ssl server definitions
temporary fix for
accept encoding missing comma add
allow to disable dhcp gateway offer when set up of ap
allow indefinite duration for tone
automatically wificlient flush before stop
allow psk instead of passphrase in wifista begin
reset artificial timezone when using newlib
scheduledinterrupts should still use current schedule_function
fix to servo allowing write to be called before attach
correct default return value for softap
don t throw exceptions from operator new by default
manually manage fifo volatility
merge pull request from shmuelzon zero content length
fix startupdate always false
fix mixup with boolean bitwise or for bssl probing
empty committ for pull request
fix puya flash write buffer alignment
leverage realloc in string changebuffer
stop lwip dhcp client when wifi goes off
disable interrupts inside esp getvcc
bool wificlient needs to return connected since clientcontext is not destroyed on stop
fix to parse parameters in the url of a post with empty content
disable debug output in serial begin
streamstring sso bug
fix wifi multi debug compile error
update wstring cpp
uninitialized stack object can have unintended bad values
fix error message typo
wificlientsecure don t send close alert when opening new session
remove duplicate return copy and paste error
initialize io_ctx to nullptr
move a couple larger strings from rodata to flash
fix typo causing crash on missing uri scheme
fix linker errors with wifimulti
post body might not be sent with the headers
only set flashmode byte when uploading an executable
if data loss exit
wificlient remoteip fix nullptr crash
fix pointer arithmetic
fix wificlientsecure connected
progmem compatibility changes to string
prevent wifi config corruption
fix serial number length
fix uart clock divider
wificlient use stream _timeout for write operations
onewire set up as input_pullup
fix missing separator between get and post arguments
remove double include of cbuf h
fix spi run on system clock
web server disconnect at the end of callback not after sendcontent
add system_soft_wdt_feed to esp wdtfeed
fix documented parameter
fix mime type for txt suffix
don t clear authentication options on a stop
pgmspace zero out memory in strncpy_p
add missing virtual destructor in transporttraits
revert disable debug on serial end if debug on this interface
add missing for the checkflashconfig call
isr check for address in iram
prevent endless loop in run_scheduled_functions
disable updater debug
remove uart register reference from hardwareserial
fix erase size in esp eraseconfig
revert revert disable debug on serial end if debug on this interface
check print write byte return and stop on fail
using std nothrow instead of malloc
allow http header value without lws
update leamdns cpp fix issue
test for streamstring sso bug
merge pull request from shrhdk allow no lws
also set timeout on already existing connections
merge pull request from dpeckett malformed uri scheme handling
re enable old behaviour if passphrase string is empty
host emulation littlefs was not fully enabled
refine log add overflow log
fix memory leak of source
fix librtmp get bytes bug
when thread terminated normally set the tid to null
fix not fix for the shrink to clear the queue
remove test code
fix verbose bug for
memory will leaks
fix the process restart bug
merge pull request from zhengfl master
fix bug of send rtmp message
fix bug for issues
fix bug fix the way calculating pcr for ts
add basic utest for config service
explicit exit st thread for gmc to analysis mem
refine annexb parse
for check the chunk header length
fix the build failed bug on centos
add comments for openssl key generator assert
merge from feilong fix memory leak
fix default value error
fix bug of auto bravo atc
for bug clear the sps and pps
fix compile failed
refine forward use utility to generate the tcurl
add comments for ffmpeg
fix start and end time bug
refine code for http remux reload
fix the config bug
for refine inotify watch for relative path
rename global username and email
cover flv decoder
refine srsudppacketrecvcycleinterval in time unit
add comments for lf cr
for improve coverage for parsing client ip
update comments for aac he or he v2
add comments for source return for sequence header
remove test code for
ci cover kernel codec
fix bug of hls close the muxer when segment finished
fix the long long warning
refine log for recv message
rename version to versions
fix fs not close when dvr error
fix bug for build failure when configure with log info
fix bug of null ptr for edge kbps object
fix bug for api level should be string
add log for ssl disabled
cleanup connections when terminate server
fix the config parse error
add comments for bug
update call response add comments
fix merge build failed
fix bug of reload ingest
for add todo for hds check whether disabled support reload
add comments for aac encoder
refine utility of cpu add comment for cpu percent
fix bug of forward erase msgs when error
add bug for http mp4
fix mem leak for send packet always free in send msg
fix the build failed bug
merge from feilong fix memory leak
for cover rtmp other codes
refine code use macro for hls resample audio
fix utest bug init fd
fix the hls bug
for bug limit the buffer for input chunk size
for remove test code
fix build failed
forget delete memset
refine code use rtmp_fmt_type0 instead
support redirect stdout by
fix bug of time jitter correcter
refine code for
fix bug for rtmp client use null ptr
fix hls muxer segmentation fault
fix bug of log message
for refine code use kbps instead skt of rtmp connection
flush the stdout buffer after logging to console
fix build failed bug
clear the queue when edge republish
fix build failed for headers for error
fix the build failed bug on centos
refine utest for tcp connection
change utest when demuxer use io interface
refine the overflow algorithm prevent smaller piece
the st_thread_exit is ok not the memory leak cause
ignore coverage for codec sps pps parsing
ci cover all kernel error
merge pull request from shelhamer dlog cudnn workspace
io cpp changed the imread flag to cv imread_color
fix comment typo
update solver cpp
make rng_ a private member of generator
fix string compare error
reduce step size in powerlayer gradient checks fix
output info from caffe cli to stderr by default
merge pull request from longjon superfluous toproto
set phase in dropout tests
check that count_ does not overflow in blob reshape
fix matcaffe printout to specify num of args now including train test phase
get back use cpu print for caffe train
use glog to provide a backtrace on crash
fix layer name in logging
merge pull request from jeffdonahue learnable param id fix
make class multinomiallogisticlosslayertest derive from cpudevicetest
set correct solver_mode in solvertest so travis build doesn t randomly
log shape dimensions for eltwise layer shape mismatch
test testreshape check small then large
oops wrong deprecation message should have
fix test data layer post lmdb
splitlayer change reshape n h c w to reshapelike
remove redundant check in lstmunitlayer
fix typo in help text for model option
test drop bogus opencv guard for layer type
fix bug for resizing images
merge pull request from malreddysid master
convert imageset comment fixup
print iteration along with every test score in solver test
fix comment typo
merge pull request from longjon check blob overflow
pycaffe live in caffe namespace instead of opening it
print to stderr for example lmdb code
merge pull request from ronghanghu show use cpu
print status message on error opening leveldb
change true_std to intended bernoulli_std
add leveldb header back to util io cpp
merge pull request from bonsaiai wrap declarations in switch
handle linter stream errors
merge pull request from alfredtofu dev
add necessary input checks for matcaffe
fix bend pycaffe to the whims of os x header order
fix lint causing travis failures
consolidaet gpu flag for caffe cli
sigmoid fix cpp
test code log
had forgotten to set some of the dataset test cases to lmdb backend
exclude core hpp when building without opencv
add comment explaining placement of system headers in c python module
merge pull request from baeuml minor whitespace in logging message fix
solver_add_nccl accepts any kind of solver
add explicit declaration does that help the flakyness
merge pull request from darrengarvey fixup memory leak siamese
fix solver indexing of output blobs was incorrect for non singleton
remove misleading comment from a test file
merge pull request from jyegerlehner preserve extracted blob shapes
hdf5datalayer die on failure to open source file
merge pull request from longjon set device early
initialize the transformer rng in the base data layer
merge pull request from shelhamer test phase fix
fix mnist comments in cifar example
revert the namespace ending comment to the same line of the bracket
pycaffe store a shared_ptr caffenet in sgdsolver
merge pull request from yuduowu master
instantiate caffe_cpu_sign for float and double
merge pull request from mnogu fix typo model option
num in blob is deprecated
figured out the bug it s curand
merge pull request from jeffdonahue pycaffe shape accessor
pycaffe build include python first in caffe tool
fix comment i forgot about from shelhamer s review of
fix memory_used_ by computing after setup
rename curand_availability_logged according to the google style guide
merge pull request from ducha aiki fix extract features
hot fix for warning
hdf5_load_nd_dataset_helper check that dataset exists first
fix bernoulli bernoulli test now all pass
merge pull request from shelhamer image fail die
disable dilated deconvolution
remove spurious constructors from caffeblob and caffeblobwrap
fix the condition prefetch_needs_rand in the imagedatalayer
merge pull request from longjon gpu ptr
update devicequery cpp
bug fix double instantiation of gpu methods in loglayer
testgradientbasedsolver drop doubled seed inititialization
pycaffe let boost pass shared_ptr caffenet
pycaffe add shape accessor
a comment misses a space char
common cpp rand seed fix
simple forward test
merge pull request from lukeyeager bvlc remove comment in tests
slicelayer fix whitespace
no need to manually delete the pointers which are managed by std vector
fix clang compilation problem w dummydatalayer
merge pull request from ibmsoe fix embedlayertest
don t end comments with so that wcomment can be used
set train in commontest testphase
tweaked gaussian filler tests for less false fails
merge pull request from willcrichton master
increment iteration during update not step
remove unused includes in accuracylayer and argmaxlayer
fix more lint
softmaxlosslayer fix canonicalize input axis
fix for solver issue pointed out by moskewcz in
remove redundant code in convolutionlayer reshape
merge pull request from longjon get layer gone
merge pull request from jeffdonahue bias gemm type fix
write create truncate prototxt when saving to fix
merge pull request from mfigurnov fix truncation warning
fix check solver prototxt parsing
fix wrong argv check for whether a backward pass should be done
fix for learnable_param_ids_
cudnn only log conv workspace in debug mode
expose params in python interface
fix include order for pycaffe on osx override lint
fix relu cudnn test bug
get rid of messy snprintf string concatenation
lock the mex file to avoid matlab crashes
fix cudnn build by readding line deleted in
merge pull request from jeffdonahue gpu flag overrides solver mode
fix whitespace error in hingelosslayer
fix flatten layer backwards to dummy return
gui resolved issue add more logic for forwarding to the search box
dbg simple logging of an instruction
dbg switch default assembler engine to asmjit
gui fix double space in infobox
dbg fix the devicepathtopath function
gui fix log string
dbg fix a bug in dbgdisasmfastat
dbg dont allow absolute symbols
dbg fix bugfix
gui fix a bug in the bridge with the reference view
dbg filter more string symbols
gui fix the disabled breakpoint color in cpusidebar
gui help on symbolic name now also works on the selected instruction label
gui system language as default interface language
dbg fix small spacing issue in window title
fix thread num in handles view
dbg fix potential bug in threadgetlist uninitialized variable
dbg properly truncate the page info
add check in jit stuff openkeyexa
dbg resolved issue fix inter modular calls
gui fix a small bug with the scroll bar range
keep highlighted token highlighted
dbg escape and in auto comments
bridge compatible with msvc
gui fix minor issue with window title for the thread name edit dialog
dbg linearpass basic_block_flag_absjmp
dbg resolved issue allow tabs in the commands
gui fix warning in symbolview
revert gui fix the disabled breakpoint color in cpusidebar
gui resolved issue copy single byte command doesn t work
dbg also set pdebuggedbase and check hash on attach fixes
change comments so they re not misleading
dbg resolved issue bug with relative symbol paths
dbg fix uninitialized variable in varfree call
dbg fix issue in _dbg_getbplist
gui better isvalidlocale function
trace record during trace
dbg allow quotes in the expressionparser
gui fix annoying things about assembledialog needs more fix with layout being too loose
should fix a crash
gui correctly break on imports dereference the address for the actual function address first
add a shortcut for copying the address
dbg dq alias for dataqword
dbg fix incorrect variale name for tracecondition
gui fix display of floats in the registerview
dbg fix visual bug in reference view init
gui properly handle little big endian in the edit fpu register dialog
gui always show patch dialog even when minimized
dbg workaround for pausing certain applications that are stuck in ntusergetmessage
gui fix initialization of structures
dbg remove useless parsetype error
gui closes issue invalid stack range when csp is zero
gui resolved issue copy selection rename
dbg gui fix compile error
dbg fix a bug with the debuggee notes being cleared on manual database save
exe skip outputdebugstring exceptions in handler
gui fix dumb compile error in assembledialog
dbg call exinfo on exceptions that are not skipped
dbg remove explicit size for memset in command line
dbg fix ignored exceptions when deleting everything the list is cleared
gui fix an out of range mistake
jump to thread ip by double clicking
dbg fix crash in stackgetsuspectedcallstack when csp is not pointing to valid memory
gui double click on cip will show the instruction pointer in the disassembly window
revert gui remove useless newline
dbg fix a bug with string detection
gui fix memory deallocation bug
delete inactive breakpoint in breakpoints view
gui resolved issue thanks to sstrato
gui consistency for non printable characters
gui remove unnecesary bar in graph image
fix removal of dllloader exe
dbg fix typo in pluginreload command
gui better view of imports module label instead of label
gui don t follow patch in disasm and dump when toggling
dbg also show thread id when the thread has a name
disabled net checking for dll
fix in memory leaks text
gui remove initial zero on the hexeditdialog
minor parsing bug
gui only toggle breakpoint when actually clicked on bullet
gui middle mouse in dump copies current selection address
gui resolved issue
dbg fix invalid free
gui fix annoying focus issue when source debugging is enabled
gui fix a bug with comments incorrectly displayed in the breakpointsview thanks to ahmadmansoor for reporting
move cursor after inserting message otherwise multi line prints are off screen until next print
dbg fix setmemorybpx command
dbg fix out of bounds access in valapifromstring
stricter ordinal name check
dbg fix getting raw int value for float typed watch points
gui fix tab order save error
gui resolved issue bug in attach dialog
command parse ignore tabs
dbg update patterntransform to reject invalid patterns
fix double hotkey d
dbg add memory breakpoint s types in bptobridge conversion
don t disassembly cross function boundary in the tooltip
gui temporarily disable cpudump column width saving has issues when changing the view torusxxx
gui select first module on updatepatches
dbg remove a check that has been proven stable in memoryreadsafepage
gui resolved issue show hide mnemonic brief now toggles correctly
gui don t highlight an invalid memory address cip
dbg fix inverted condition when reading global notes
gui show label of memory address in cpuinfobox
dbg fix stringutils trim
minor fix propagate mousemoveevent to hexdump to keep multiple data selection still possible
dbg reduced memory map refresh time
gui better focus stylesheet for cpu view
dbg addressofnameordinals is word not dword
don t convert to in browse dialog
gui add zword
gui fix bug if you execute msgyn and then an error
dbg fix issue
gui fix the weird compile error
gui make sidebar dots bigger
dbg small fix for modules without extension
gui small fix with comparing tokens
fix export name parsing
patch list in separate thread
dbg use async send in msgqueue
dbg fix a bug in memfindpattern thanks to computer_angel
gui fix issue editing expression using gui changes the watch type back to uint
remove backtab key substitution
dbg refresh views in commentset
gui fix copy selection
fix a bug when updating disassembler config in trace viewer
zydis fix uninitialized variable in bytesgroup function
auto move cursor on toggling bp in reference view
dbg remove deprecated for memory location
dbg fix log command glitching the status bar with no parameters
hash changes reverted as per irc chat
dbg remove print for program is already running
gui yet more unusual instructions
dbg clarify fix something in linearpass cpp
dbg readexportdirectory put upper bound on the number of imports
dbg remove n replacement in string formatting
gui apply unique color to unusual instructions
dbg acquire ownership of message stack before freeing it
gui fix non addresses in the reference view
gui don t update the freeze stack action on every debug state change
dbg fix typo
fix flag setting when flag is already set
dbg explicitly handle null types in scmp
fix transparancy in the stdicontable
dbg nothing acturally changed
dbg allow plugstop to indicate freelibrary should not be called
gui disassembly preview on ret instruction
remove unused variable
gui fix flags set in registersview
allow ctrl middlebutton to copy rva
fix zero termination
dbg temporary fix for autopatchexporter
fix resolving symbol size always returning true
gui fix memory map focus on follow in memory map
gui fixup now buffer instruction is used with the isselected method
do not add breakpoint if setbpx fails
gui enable search in current function menu
dbg fix a compile error
dbg conserve memory when copying bblock vectors
dbg add breakpoints table
dbg fix minor issues with mnemonichelp
gui resolved issue stack segments being displayed incorrectly
gui fix crashing issue
dbg unregister dbghelp callback when doing dbghelp cleanup might have to do with issue
gui also set the log font in the qcompleter
dbg speed up with waiting for threads to close
dbg fix possible widechartomultibyte crash
dbg fix an error with bookmark enumeration
minor fix no need to free a null json object
gui nothing actually changed
dbg fix command line parsing closes issue
gui fix warning in hexvalidator
dbg small change to the findall command to support module search
gui fix width of hex short
dbg fix invalid report of assembled branch does not point to an executable memory region
dbg fix a bug with detaching while paused sometimes
fs gs segment support in dbggetbranchdestination
dbg remove debug output
gui keep consistent when disabling summary closes
fix filename bug in patch menu
prioritize private symbols over public
bridge fix a terminating error message when there was no settings file
gui resolved issue reference view does not follow in disassembler correctly
gui get rid of any message boxes
gui fix creation of dll breakpoints with a space
gui handle highlighting segment registers in registersview
dbg benchmark for disasmget
dbg fix a typo
gui close bridgeresult event handle
dbg fix a bug in escaping strings
project simplify launcher code even further
gui remove tst from referenceview
dbg fix a crash when loading old database files fixes
gui astyle cpustack
dbg fix a bug with breakpoint count not shown
dbg dont save database when there is nothing inside
dbg faster disasm instruction
dbg fix locking condition
gui no more uppercase module names
a little fix in setpagerights
gui don t display xmm ymm st registers in the infobox resolves issue
gui close issue invalid sizes in save data
tolower for a better of the old instruction
gui allow esc and backspace in the shortcutedit
hide overview widget now set appropriate splitter sizes
translate this file
dbg set foreground on cbsystembreakpoint
dbg remove redundant null terminator assignment
bridge macro fix
dbg dynamic comments works like logging breakpoints format string
dbg add thread suspend failed message in threadsuspendall
fix writeuints bug
gui fix a bug with the tabwidget
gui reset hit count is working again
gui actually fix the crash derp
dbg resolved issue stupid i didn t implement it this way in the first place lol
do not set ntterminateprocess breakpoint per default
dbg resolved a potential crash when calling cbplugin functions
gui don t check queryprocesscookie per default
dbg remove debug log statement in value cpp
dbg fix a small typo
dbg remove a redundant log statement from the patch process
gui changed default color for unusual instructions
add helpful placeholder in command dialog
gui allow disabling of seasons
cpustack freezing the stack no longer incorrectly causes the user to be unable to select a different address
dbg small sprintf optimization
dbg fix log instruction
gui update the source view if the disassembly updates
gui double click in memory map to follow in disassembler
dbg add forgotten cbselchanged in pluginload
dbg fix an invalid handle exception
gui fix issue trace record in popup works again
dbg updated valapifromstring
dbg fix issue incorrect detection of unary operators
fix for follow in dump n in stack view
gui correctly report function size to snowman closes
dbg fix artificial build error
set window focus when toggling with graph detached
exe fix potential conflict with time stamps use milliseconds
gui updated default auto comment color
dbg updated scylla it will now automatically use cip as the entry point which is almost always what you expect
created additional shortcut to the copyaddress function
dbg remove restrictive heuristic for string detection
dbg fix incorrect efree in command
gui conditional tracing instead of rtu in run to user code
gui fix bugs in guiselectionget guiselectionset thanks to ahmadmansoor for finding
gui replace backtab with shift tab in shortcutedit
merge pull request from mrgreywater fix debug string termination
dbg add reason to efree
dbg stringformat fix
gui no bom in redirect log
gui fix two warnings
dbg fix a crash in plugins on cb_breakpoint
gui fix compiler warning
dbg use patterntransform in the find command
dbg fix a crash with very big commands
gui possible fix for an irreproducible crash
gui hide main window on close
fix timer leaks
update stringutil cpp
gui add asterisk for log link color because it cannot be applied to old contents
gui more realistic scroll bars
dbg vector clear is redundant right before destructor
gui ignore ret branch in disassemblypopup
dbg more message fixes
be specific about which string_view
add null check in uimanager updateview
playground app not specifying the uidispatcher
rethread after resuming timeout coroutine
turn off debug output for batchinguimanager
remove default padding on flyoutpresenter
fix fast refresh in playground app
fixes to formdata network support
small fix in the c dump code
fix test to pred
change default distributed mode to row
change include order so that rinternal does not disturb us
adapt tree booster
chg r package path back
change test task to pred
add deleted main back
jvm packages catch dmlc error by ref
remove some stray bits of the reaper_thread
watchman remove unused includes
fixup pending list linkage mutation
fixup windows compilation
move markdirdeleted from handle_open_errno
propose fix one typo
log when connecting to unix socket fails
avoid crash when loading triggers from the state file
watchman relax fuse inotify watch restriction
watchman mostly fix mode mac builds
watchman remove asprintf from info cpp
delete operator for subprocess options
suppress rusage deltas from watchman perf logs
add keep alive support to manualexecutor
never attempt to monitor fuse filesystems with inotify
remove pthread usage from checksock cpp
disable telemetry logging for scm commands
codemod future then continuation lvalue ref
don t leak client states hash
fix watchman cppclient test
fix population of query context in benchmark
use symbolic name for the filename component of the inotify buffer
have the view trigger a file watch
watchman assume we were built with cmake in childproc test
watchman improve debug logging for slower commands
fix off by one for wide string size on windows
fix facepalm with posix spawn
future t then future t then not try task future t thenvalue task
fix debug level for json input
remove unused includes in watchman root init cpp
move watchman_state_file dont_save_state globals out of main cpp
small dependency cleanup
fix potential dangling reference
issue update warnings to add appropriate quotes
watchman fix null defref in client mode on windows
avoid reference cycle for subscriptions
watchman migrate mercurial test to gtest
potential fix for deadlock when unsubscribing in watchmanconnection
subscription default defer behavior change
defer shutting down configuration
add missing singleton shutdown handling
fix watchman to call folly singleton s registrationcomplete
allow building watchman on macos
allow line images with larger width depending on height
merge pull request from stweil limit
fix comment which referred to unused tesseract parameter
report when tesseract legacy engine not present fix issue
fix compiler warning wtautological undefined compare
remove unused variable
bah fix this properly
fix some typos in comments found by codespell
fix insecure call of tprintf
copy resolution of source image fix issue
catch std runtime_error exception when setting the locale in debug code
correct noimages in debug pdf file
fix lambda function for curl code errors
merge pull request from stweil scanedg
deleted lots of dead code including pblob
add missing char from format specifier in tlog
merge pull request from stweil master
fix assertion caused by wrong unicharset
choiceiterator skip symbol without choices
ccmain paragraphs fix memory leak
osresults fix runtime errors detected by undefinedbehaviorsanitizer
opencl fix mismatched new free
sw update build script
missing close bracket
add lstmbox and wordstrbox to c api
fix compiler warning wunused function
update sw build
classify adaptmatch fix memory leak
update sw build
merge pull request from stweil crash
autopageseg findblocks gridremoveunderlinepartitions avoid self deletion
use ndebug macro instead of debug
simplestats remove unused method
fix two typos in comments
add missing cerrno includes fixes
tesseractmain remove unneeded duplicate code
update capi cpp
merge pull request from hsen dev master
typo correction specific
ccstruct fix compiler warning disable buggy code
add necessary intproto h header to protos cpp
add more missing include statements for windows build
remove unused extern macros
classify remove unused constant kminpclengthincrease
merge pull request from stweil cmdline
wordrec fix compiler warning wstringop truncation
merge pull request from stweil scrollview
fix using bilevel tiff in pdf output
baseapi simplify code
identify all supported tiff version by leptonica
fix windows build
dict dict fix memory leaks at program termination
fix typo in non vs build
fix a typo in the c api progress monitor
fix tesseract exe not flushing stdout stderr issue
sw do not build unit tests by default
merge pull request from noahmetzger coverity
merge pull request from stefre tsv fix
merge pull request from stweil warnings
fix sw build after upload
api baseapi fix memory leaks at program termination
fix debugwin path
lstmtrainer fix diagnostic message
findandremovelines findverticalalignment decrease fix vline min length
merge pull request from stweil lstm
chop use more efficient float calculations for sqrt
don t trigger a deliberate sigsegv for fatal errors in release code
strcasestr needed on cygwin too
merge pull request from stweil fix
add additional unicodes to isvedicaccent
lstmtrainer set constant klearningratedecay at compile time
only print merging rows in debug mode
add multipage tiff capability
fix reading of parameter from traineddata normproto component
fix sum computation in higher precision
fix bug that was introduced with graphics_disabled changes
fix invalid f spec error while building fullyconnected layer
classify modernize function uniformdensity
merge pull request from stweil locale
add parenthesis in boolean expression
raii pdfrenderer cpp cidtogidmap
lstm small code optimisation
intsimdmatrixsse remove unused include statement and simplify code
lstm fix possible float division by zero
allow to have text file with list of images as input
merge pull request from stweil pi
fix bug in max_max_dist
fix windows stdout messy code
merge pull request from shreeshrii pages
a minor accuracy improvement on punctuation
merge pull request from stweil master
merge pull request from stweil msvc
commandlineflags fix compiler warnings signed unsigned
provide info about compiled openmp version
cube fix coverity warning caused by unneeded null pointer check
better fix for re enabling training
fix the memory leak double free cleanly
merge pull request from stweil formatstring
use deprecated pango_fc_font_get_glyph for old pango versions
merge pull request from stweil leptonica
fix empty statements
remove unused iterator
use cstdio instead of stdio h
destroy box before potentially exiting function
fix use binary mode when generating pdf to stdout on windows
add missing ifdef in tesseractmain cpp
merge pull request from stweil fix
revert debug message commited by mistake
blockpolygon clip to image rectangle
add missing argument for tprintf
remove duplicate include fixes
update comment about format
sw try to fix build
fix broken build for windows
fix bbox as the entire page
merge pull request from stweil lstm
fix regression of issue again
merge pull request from rmtheis fix hocr title
merge pull request from stweil valgrind
wordfontattributes check that word nullptr earlier fix
merge pull request from stweil lstm
fix indentation of hocr output
remove unnecessary conditional
restore vc compatibility for paragraphs cpp
ccstruct remove unneeded null checks
add kvedicmark to consumevowelifvalid
lstmtrainer catch empty vectors
report libarchive support
remove unused variable
fix windows build
training fix compiler warnings deprecated register keyword
update src api tesseractmain cpp
opencl fix type of parameter for clgetcontextinfo
api remove unused variables
fix build with disable graphics
fix syntax error regression
merge pull request from mingodad mingodad fix interword spaces
more code cleanup from patches and fix warnings
fix name collisions mostly with stl
fix imagethresholder otsuthresholdrecttopix for opencl
update imagedata cpp
fix debian bug assert failure during training
fix typo and add todo comment
fewer g wunused variable warnings
remove assertion which is triggered by tests
fix potential crash in string class
scanutils fix illegal memory access
fix new bug in error message
add missing namespace prefix fix build for merge_unicharsets
remove unused api function tessbaseapidetectos
merge pull request from solomennikm fix reading carriage return in config files
scrollview clean include statements
fix start scrollview jar failed when lstmtraining
merge pull request from noahmetzger master
fix text box width hight calculation
ccutil tessdatamanager fix resource leak
fix crash if output file could not be opened
fix cygwin compatibility part ii
fix crash in preloadrenderers with nullptr outputbase
cutil replace true false by true false
tesseractmain fail if bad command line option is given
merge pull request from stweil format
fix null pointer dereference when writing font name into hocr
avoid buffer overflow issue
fix text box width hight calculation addition
merge pull request from shreeshrii fixvalidcodepoint
change page to line in message
don t use null for integer arguments
remove unneeded include statements for pgedit h
add terminating linefeed to fonts conf
merge pull request from stweil cid
merge pull request from tdhintz tdhintz stdmax patch
lstm remove unused variable
fix syntax error
clean use of double float
improve formatting of hocr output with character boxes
remove obsolete code
fix closing meta element in hocr output
use lept_free to free memory allocated by leptonica
ccstruct remove unused local variable
remove duplicate include statements
remove unused includes
merge pull request from stweil lept
fix indentation after conditional wmisleading indentation
unicharset fix compiler warning signed unsigned mismatch
merge pull request from stweil fix
deleted lots of dead code including pblob
fix compiler warning
remove dummy code from lstmtrainer inittensorflownetwork
main fix two memory leaks
change type to uchar32 to fix isvalidcodepoint
fix problem with preadapted templates
fix issue reading older style page number less box files
merge pull request from noahmetzger winfix
fix access to uninitialized variable
merge pull request from stefre tsv fix
fix some compiler warnings unused local variables
fix argument for tprintf
what the git s merge failed hard
build sw disable fma dotproduct
fix issue detected by coverity scan
move comment about swap meaning for deserialize to correct function
merge pull request from stweil params
merge pull request from stweil fix
merge pull request from stweil fix
fix constructor for class dict uninitialized member variables
fix first word only bug in pdf output
sw depend on leptonica master
fix an error in setting char properties
fix an issue where we sometimes leave a dangling outline loop pointer
lstm initialize member variable input_width_
fix intraword spacing for slightly better pdf copy paste performance
merge pull request from stweil errhandling
merge pull request from noahmetzger winfix
update for get_pid_directory
fix test_case name
load kexts at launch
update appendix console_user_server_client
add log message
output log message to std cerr
update mask appendix eventtap
apply clang format
call manage_session_monitor manage_console_user_server at start
add filesystem is_directory tests
remove blank line
update appendix version_monitor
initialize dispatchers after another process check
initialize dispatchers after another process check
wait until session get_current_console_user_id is success
set sig_ign for sigusr1 and sigusr2
use pqrs gcd scoped_running_on_main_queue_marker
add log message
remove unused include
fix multiple macro definitions of glfw_include_none
merge pull request from adamdmoss glfwcursor
nav fix navdown to open menu from a menu bar
tentative gcc clang warning fixes with unused functions cf
tables remove dubious window skipitem assignment in tableautoheaders
nav hovering menuitem selectable with mouse makes navid matching what seems to be windows beahvior at least for menus
nav remove unnecessary test always failing
nav navprocessitem updates current navlayer basically we re hearing toward rebuilidng nav info from an id
selectable render into autofitpadding but do not extend it fix small visual gap
style enable window border by default
examples vulkan fix missing subpass dependency
fix a memory leak of imguicolumnsset s columns vector imvector doesn t call destructors
fix wconversion warning
examples vulkan tweak
fix unused variable warning
carriage return in demo tooltip
merge pull request from orbitcowboy master
aa branch notes
imfontatlas if stbtt_packbegin because of failing allocation we return a little more nicely keeps coverity static analyzer happy
fix incomplete replacement of unsigned short with imwchar in interface of imfont findglyph
examples ocd tweaks a sign that we are near the release
fix uninitialized fields mousecursor set by first call to newframe
popups calling openpopup on already open popup doesn t close it s child
demo fix incorrectly formed string passed to combo
merge pull request from bkaradzic master
metrics window shows indices along with triangles count
separator within column correctly vertical offset all columns content pointed out in
begin remove asserts that got in the way of some flags combination
fix space issues
menuitem tweak to not draw over all horizontal space when in horizontal layout mode
add comment on using pushfont before building the font
columns fix previous commit wrong chunk commited
slider fix hovering bounding test excluding padding between outer frame and grab
popstylevar fix oops
remove inline marker
don t do a bunch of os gymnastics alloca h is only for glibc everyone else uses stdlib h
showtestwindow add a custom label in the format for one of the slider example
implicit debug window uses a debug default identifier to allow the user creating such window with custom flags
imfontatlas addfontdefault made it possible to override glyph ranges
console example fix losing text selection when using tab completion with no match
newframe prevent division by zero in frame rate calculation if io deltatime is continuously zero
child window explicitly inherit collapse state from parent
nav honoring imguiitemflags_nonav which is used by color picker
renamed section in documentation
settings make imguiwindow settings always first in the list
fix columns lines not being pixel aligned
windows no default ime handler when compiling using gcc
nav remove old unused code
bullet fix inconsistent layout behaviour when clipped
dragfloat double clicking turns into an input box along with ctrl clicking
remove spurious assert
treenode node with the imguitreenodeflags_leaf flag correctly disable highlight when dragdrop is active
aa branch fix input cursor
fix colorbutton showing wrong hex value for alpha
context newframe asserts
nav moving window with nav stick disables mouse hover
merge pull request from pagghiu bugfix
merge pull request from develophant master
fix for correctness and allow non auto resized modals to have window minimum size applied hopefully no side effects
fix inputtext bug with imguiinputtextflags_enterreturnstrue in nav branch only thanks grouflon
fix compile issue bloody git stashes
selectable fix misuage of getcontentregionmax x
fix to facilitate branch merges
updated example code
remove unused parameter in demo window code
demo tweak irritating pink color
a tiny optimization to imlineclosestpoint remove unnecessary sqrtf call
nav allow padleft to close a menu
vulkan fix not incrementing semaphore index
selectable fix text clipping
initial creation of a imguiwindowflags_nobringtofrontonfocus window push it at the front of global window list
active widgets is not clipped so it can always release its active state mentioned in
fix minor text clipping issue in window title for when using font straying above usual line
internals endgroup remove unnecesary parameter to itemsize
improve warning configuration for clang
examples remove unused variable will be used in docking branch tho undo when merging
examples sdl fix minor warning
minor optimization so that mass calling beginpopupcontext functions can early out more commonly without hashing id
silence borderline warning with werror strict overflow
selectable variant fix missed return value
imdrawlist channelssetcurrent add an extra assert
example fix popstylevar on vertical slider to avoid assertion
fix font config propagation in fontfrommemorycompressedttf
progressbar fix rendering when straddling rounded area
add assert to catch users using begin end without a first newframe
argh remove redundent defines
nav calclistclipping imguilistclipper fix to ensure we can navigate clipped lists
fix unused variable warning if asserts are compiled out
backends allegro don t call addinputcharacter if the pressed key has no character
internals splitter behavior doesn t show a border
tables fix stacked popups incorrectly accessing g currenttable of parent in stack windows
fix unaligned memory access fix font loading when building on emscripten
dummy create an item so functions such as isitemhovered can be used
fix nav branch merge issue
warning fix in demo code
imfont calctextsizea fix font scaling with fallback character
demo satisfying buffer size
fix openpopup accessing current window effectively opening debug on an empty window stack
demo console fix a completion bug when multiple candidates are equals and match until the end
remove non text character at start of imgui cpp
delete merging artefact that survived for a day
fix warning missing initializer for member
inputtext add comment
demo displaying version number in test window
inputtextmultiline fix so that isitemactive can be used afterwards otherwise the info was lost by using child group
fix rendering of histogram with a lot of values
fix enum typo
nav fix popup wrap around logic for windows with scrolling
add assert in treepop to ease catching id stack underflow earlier
speculative build fix for old versions of visual studio compiler
nav remove an unnecessary assertion that could fail when using mouse and keyboard simultaneously with the navenablesetmousepos flag
demo inputtext password demo filters out blanks
silencing wstack protector
fix warning missing field w initializer wmissing field initializers
inputtext fix ignoring text inputs when alt is pressed
fix warning when imgui_define_math_operators is already defined by build system
imfont calcwordwrappositiona minor optimization
style made scrollbars rounded by default because nice
tweak context menu example
aa branch fix
imfont fix minor bug with cpu side vertical clipping of text
examples sdl opengl explicitly setting gl_unpack_row_length to reduce issues because sdl changes it
dragint dragscalar actually made the fast past handle the now most common case
fix undefined behavior
separator logs itself as text
set pixel shader and vertex shader to null
merge pull request from roflraging master
use fabs instead of abs to ensure we use floating point
tweak expression to be less weird how did that ever happen
internals newframe shuffled some code around to minimize upcoming patches
showtestwindow minor tweak
fix signed types warning in pasteboard handler
silence xcode static analysis false positive
tweak default texture width for large amount of characters
fix new windows auto fitting bigger than their ini saved size
tables non resizable columns also submit their requested width for auto fit
navinitwindow change imgui_debug_log to imgui_debug_log_nav
collapsing header also honor framerounding
tabs fix support for drag and drop imguibuttonflags_pressedondragdrophold incorrectly missing from the merge from docking branch
set required depth member for buffer image copy
imguitextbuffer appendfv needs to va_end the args_copy
fix file descriptor leak if imbitmapfont loadfromfile calls to seek tell fails
splitterbehavior not using framerounding in render was in first commit of the function not sure why
fix cross compilers
listboxheader can return false also made listbox only use public api
alloca fix to allow clang with microsoft codegen path
nav calling setitemdefaultfocus doesn t make mouse cursor dirty if nav highlight is off
add an assertion
fixes warning caused by a missing switch case
speculative va_copy fix for old versions of gcc clang cannot repro
comments fix missing line in the how a simple rendering function may look like section
fix uninitialised fields in imbitmapfont were unused when uninitialised but still dodgy
setscrollposhere fix imprecisions
imfont rendertext faster handling of rendering above clipping rect useful for
selectable not specifying a width default to using max of label width and remaining width
autofit cancel out one worth of vertical spacing for correct symmetry group and tooltip already do that
merge pull request from crunkle master
fix using isitemhovered after endchild
tweak button sizes
fix compile warnings in sdl examples
inputtext when keys are being input while mouse is dragging disable mouse drag until button release
examples sdl fix mapping of insert key fix bug introduced in
examples clang warning fix
demo layout remove unnecessary beginchild endchild calls
internals bringwindowtofront iterate front to back faster in majority of use cases
using getcurrentwindowread instead of getcurrentwindow
inputtextmultiline multi line selection draw fix
minor fix to avoid undefined behavior sanitizer triggering
buttonbehavior fix imguibuttonflags_noholdingactiveid from incorrectly setting activeidclickoffset which probably have no known effect but it is more correct this way
aa branch collapsingheader rounding down half window padding
nav inputtext fix using up down history callback feature when nav is enabled
typo elif instead of else fix
fix minor error in doc
merge pull request from waywardmonkeys improve clang warnings
getcoloru32 imguicol avoid using gimgui twice since some implementation make it a tls ish variable with non trivial accessors
imdrawlist better looking non aa rectangle lower right corner and rounding
nav debug overlay uses default font
i m not ok
udev fd check
per owner monitoring
shell ensuring shell does not access stop event
restrict permissions on rocksdb paths
merge pull request from alienvault labs master
no uuid uuid h under freebsd
fix ubuntu version regex
ignore osx warning
fix restore balance to the double force
vtable fix warning for process in condition assignment
create copy of uuid data so that we have a value and not a reference before releasing the properties
check for nullptr in createpropertyfromcertificate
fix processes table to report gid correctly
add missing includes to posix filepath cpp
fix the shell text skip ci
string length instead of strlen
locking issue with osquery resetdatabase
extensions clear signals in autoload thread
rocksdb remove manual flush
fix process free regression for libprocps
fix the last strcpy
const reference iteration of kdomains vector since it s const itself
filtering out lowbattery crash logs
tests remove username test in favor of status check
carver emit an error if the carve function is disabled
merge pull request from kaizensoze small temperature value change
move rocksdb logs to info
set global output function for thrift
fix osqueryd ignore pidfile parsing errors
flush stdout with help
handle expect in test
add check for character_frequencies size
disable query caching in tlsserverrunner
add status back instead of query data again
log files to disk close
freebsd avoid clash with rapidjson
tests improve flakyness of test_processes_memory_cpu
treat iokit hid failures as warnings
add osquery version to status logs
tidy fix syscall deprecation on macos
fix init when using fbthrift and folly
support date type in os x preferences table
merge pull request from facebook glog to file
fix error when wrong directory separator was used
look up the external plugin by the correct name
fix use after move in yaraeventsubscriber configure
restrict regular file checking of tls pinned cert to windows
fix lock resource leak in events syslog
merge pull request from mlw fix lingering socket fds
correct config loaded meaning to be has run load
fix usb class subclass ids on darwin
document logger_mode flag takes decimal values
fix a typo in a variable name
gracefully shutdown auditdnetlinkparser will not wait indefinitely
freebsd fix patchlevel reporting
tests improve hash table integration tests
prevent extensions autoload when extensions are disabled
daemon set service status to service_stopped on exit
was closing the db in the middle of the loop instead of after it causing subsequent queries to fail
changing init verbosity to honor verbose flag
fix potential leak
aws fix tsan warning in request exception
small equivalent change to temperature calculation
comment out unused parameter to make linter happy
unref udev monitor during events teardown
change fgetln osx specific in favour of getline both osx and linux
extensions fix path quotes on windows
set active distributed plugin within extensions
allow tls to proceed without node key if enrollment disabled
manual fix to spacing issue
add path to error message when config file not found
fix up mdfindtests check for the number of columns in query result
use namespace instead of define for boost filesystem
move yara relative paths to etc osquery yara
fixes compilation issues with latest master branch
default logtostderr to true remove logstderronly
catch exceptions by ref on windows processes
ux add verbosity if osquery fails to read system uuid
properly intialize bufferedlogforwarder for tls output plugin
log event results as a flat map
uptime implementation on windows
tables augeas remove trailing slash from files prefix
fix emit warning log with unknown options
fix fix constraintlist literal_matches function
fix bug when checking if config is valid
check for interrupt requests in tls config refresher
update hash cache inode number
merge pull request from kaizensoze remove some temp related comments
fix ctor logger locking
fix ipv6 socket s family in socket events
fix the memory leak in the macos certificates table
iostream in shell
fix rocksdb crash
merge pull request from marpaia host identifier
do additional flag incompatibility check
revert small db optimization
prevent error when tls response body is empty
fix aws credentials test failure when env vars set
fix getproclist indexing
remote fix output size for tls compression
unused code after refactor
merge pull request from donhusa time
fix test on high sierra
fix warning with constants initialisation in magic cpp
change the registry like path constraint to filter recursively
enable db in osqueryi when database_path specified
remove unused unused mutex config_valid_mutex_ from config
more appropriate logging controls
fix crash in fim processing on linux
fix error in run when tables emit blank columns
skip daemon initialization if checking config
fix memory leak at osquery devtools shell cpp
merge pull request from arirubinstein master
add canary path on empty fsevents subscription set
fix update of event plugins when config fails to load at startup
merge pull request from sahalsajjad master
get rid of alias verbose_debug and debug
remove useless include of iostream from status h
fix error loading packs not verbose enough
http move away from deprecated async_resolve overload
vlog instead of warning
watchdog fix disabled database and logging when watchdog is not used
use safestrtoll instead of strtol
fix do not lock kattachmutex within shell callbacks
add timeout to netlink socket read
fix sign compare warnings in tests
improve event expiration mechanism
fix flaky loggertests test_logger_log_status
fix typing issue with cfnumber conversion
fix regression in calling core only plugins from extensions
changed logic to ignore when history file is not found expected
fix call to localfree on uninit ptr
fix rapidjson error asserting in configuration
add warning line for registry queries against hkcu
worker prefer null character over spaces when clearing argv
limit the number of rocksdb log files
grabbing additional data fields for windows event logs
populate path correctly in firefox addons
include proper headers for freebsd
attempt to query platform uuid on linux
make linter happy remove whitespaces
merge pull request from pickmansec master
adaptive listgen block dim on gpus
fix non pot particle size nan issues
vectorized ray_march on cpu
fix all legacy tests
bug fix master due to compilation error
fix statement replace bug
fix windows bitcode loading
fix clang build
passed all cpu tests
multiblock cuda struct for
another svd issue
space before thread id
global store grad update
ir don t do optimization for jit evaluator kernels
gui support mouse wheel event on mac cocoa
snode writer without unified memory
opt out useless casts
fix global store type check warning
add eval to mpm data loads
fix jit failure by setting accessor functions private
compile time storage range
mpm3d working on turing
misc fix type ownership in python scope
damn std tuple does not arrange elements in order wtf
remove debugging statement
adaptive mem size
emergency fix for non gl build
damn finally works
test suite starting point
depth limit slider
reproduce mpm slp bug
fix print_traceback overflow on os x
full matmatmul tests
fix pure cpu compilation
const fold mul
do not demote dense struct for with bitmasks
fix non os x builds
fix cg crash with zero initial residual
capture cocoa mouse move
reproduce tid mpm3d crash
cuda llvm module broken is ti_warn instead of ti_error
opt eliminate whilecontrolstmt with non zero const conditions
disabled vectorization for ci
interesting using range for fixes the issue might be a bug in listgen
remove weakened test cases
fix amal without fonts
fix non cuda build
reproduce heisenbug caused by die
fix non gpu build
larger particle bin
print task list length
fix windows build testing cpp
try fix travis
cleaned debug output
async record the kernel s name in the hash only if it has arguments
reverse os x error traceback to make it the same order as linux
misc fix compilation warning
trash unused return value
fix infinite signals
fix windows and os x build
final mpm benchmark code
remove the counter that limits slp passes
reenable llvm after fix glibc abi version issue on ubuntu
fix non os x build
misc split accessor and other kernels in statistics
fix cnn compile
