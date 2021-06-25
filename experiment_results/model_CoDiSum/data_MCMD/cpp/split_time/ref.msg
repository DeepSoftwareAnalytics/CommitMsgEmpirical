add size warning to viewport node
merge remote tracking branch origin master into master next
merge remote tracking branch origin master into master next
fix bad string name of options
fix signed unsigned mismatch preventing configs from surviving
clarify the viewport size warning message
test use fs namespace in dbwrapper unicodepath test
unblock waiting mutations
not register pthread_atfork in child process
fix windows setenv check
fix query hang
merge remote tracking branch origin master into master next
c fix uses of old configuration names
fix clang tidy check
fix clang tidy check
add catch all for completeness
run default signal handler after custom signal handler
gbmbufferobject guard against creating if we already have a valid fd
cgbmbufferobject set m_bo to nullptr when destroying the buffer object
gbmbufferobject close fd in destroybufferobject
vcpkg move package abi access after they have been computed
fix lambda function for curl code errors signed off by stefan weil sw weilnetz de
style check fix
fix pulledversion of backup worker not sure why the cursor s version can be smaller than before
remove new lines
remove redundant header
update partlog cpp
doesn t look like i ll ever complete this
remove global scope from sigprocmask etc these functions may be macros on some platforms e g mac os
update shellcommand cpp
register poco odbc connector on clickhouse odbc bridge startup
use of previously declared variable
fixes navigation path reset
undo code format
invoke codeformat on inbox changes
disable auto tmm when not using default savepath from monitored folder
handle nil with any number of parentheses
remove incorrect comment
extra initialization for msan
backupworker remove assert in pop
remove settings from query
fix incorrect shader block parsing
add missing space in storagedistributed logging sharding key
update status bar and mouse cursor when editor scroll changes
contextmenu hide favourites option if we have no path
remove files from part correctly
gardening remove unused astcontext variable
nfc remove duplicated stats tracer
fix incorrect comment
objdetect qr avoid bug with empty input
favourites dialog fix crash
remove some dead code from printast
rpc add missing helpexamplerpc for getblockfilter
moved pattern formatter from spdlog details to spdlog
moved pattern formatter from spdlog details to spdlog
windowing gbm register using cudmabufferobject
make booleans const
merge remote tracking branch origin master into master next
update storagemergetree cpp
update altercommands cpp
testing only with ok documents
reformat range based for loop
update maxsubarraysum cpp
u date comments in maxsubarraysum cpp
add missing qpainterpath header include
autocomplete for load function implemented
fix bug in for condition
add small log message
fix buckets polygon index bug
windowing gbm register using cdmaheapbufferobject
update vntaptd cpp
pvr cpvrepgtagscontainer gettimeline fix edge case where a gap tag needs to be appended to time line
add blank line at end of file
update code style
fix expression transform for empty totals
remove accident change
check for overflow when calculating sum of outputs
update unit test
update storagedistributed cpp
update unit test
count the unique number of tags in startedbackupworkers
include simulator h to restoreworker
assert recoverandendepoch does not become ready
add removal for backupworkercache
correct comment based on review
fix assert failure should be backup s contiguouslogend maxrestorableversion
remove useless code
stop actors for displaced backup workers if the worker is displaced it should not update backup containers
remove a spurious assertion it s possible that there is a gap between backup s contiguouslogend and snapshot version
a fix in appreciation of ubsan
remove memory allocation in inner loop
fix incorrectly marking a backup job as stopped this causes missing version ranges for mutation logs
merge remote tracking branch origin master into master next
update interpreterinsertquery cpp
bugfix incorrect erase element
bugfix show chat description
update weak hash
update weak hash
update weak hash
update weak hash
fix pvs warnings
don t recreate tray icon if it is disabled
make search in richtextlabel case insensitive
atomicparallelrestore should not return a version
add missing transaction reset call
csdiagnostics handle optional inout to typed pointer conversion diagnostics on generic mismatch
update pointinpolygon cpp
add a note about double escaping in regexes
force read in order of parts using single thread
fix flaky unit test
fix a crash failure
fix clang tidy
start lookup outside of d when d is e g an extensiondecl since the attribute is outside of it
windowing gbm gl register using cgbmbufferobject
windowing gbm gl register using cdumbbufferobject
windowing gbm gl register using cdmaheapbufferobject
windowing gbm gl register using cudmabufferobject
windowing gbm gl register using crprendererdma
add code for coins problem of codechef in c
changed function name and updated file path
drop old comment
updated the vector declaration
remove parameter n
remove redundant restorerequestdone break in backup actor
tweak eeprom storage type
fix macos compile error
solve indentation issue
fix some spacing
made few changes
take trace mutex in setloggroup
update readbufferfromkafkaconsumer cpp
update checksortedblockinputstream cpp
update storagedistributed cpp
fix tidy check
fix macos build failure
try fix tests
amend fix string like initial implementation
type checker remove dead code to compute pattern checking options
fastrestore load range and log files in parallel for new backup format
update basedaemon cpp
fix regression in updaterow method
minor compilation fix
ros stop rebroadcasting static frames
merge remote tracking branch origin master into master next
initialize struct variables clang static analyzer was emitting some warning about uninitialized variable usage and this patch fixes it
update main cpp
update vntaptd cpp
try fix tests
fix freebsd build
replace outdated header
update strwn cpp
gui handle walletmodel unload asynchronous this change prevents deleting a walletmodel instance while it s being used
turn on polymorphic parts for log tables
filter only chats from the list in exceptions
fix switch to filter in single column mode
don t close filter by escape
fix crash in forwarded dice media
fix fakeunreadwhileopened flag reset
reset chosen filter on escape
fix layout of cloud themes in a narrow window
allow emoji suggestions in filter titles
fix closing filter by escape
fix via bot information in stickers
allow deleting images in documentdata
use animated emoji size for the dice
close media viewer when showing file in folder
gardening use std isnan
try fix tests
shader lea fix op_a and op_b usages they were swapped
cs fix a locator path element this locator is for dictionary s value generic parameter
handle clang targetinfo hexagonbuiltinvalist
fastrestore remove assertion on mutation sending order
update regexprowinputformat cpp
update systemlog cpp
yuzu fix the stuck in fullscreen mode bug co authored by valentin vanelslande vvanelslandedev gmail com
register function arrayreduceinranges
update arrayreduceinranges cpp
update arrayreduceinranges cpp
clean up code
lua support to use image drawsprite sprite without frame to render the first frame
rpc remove unused getbalances code
fix marks count for non adaptive parts
add an assert for probably dead code
merge remote tracking branch origin master into master next
fix remove pinned dialog from folder
don t accidentally close edit folder box
modified by remove namespace using directive
resolve compilation error
ros fix skip empty images
fix a race between ddlworker and ddlworkerclnr during startup that causes ddlworker to exit because of zookeeper nonode error
astverifier nfc remove verifybound this isn t used for anything
don t use default title if it doesn t fit
fix possible crash in inconsistent historywidget state
increase get info pririty to medium high
add omparrayshaping to a few switch statements
fix popupmenus letterjumping on any inputevent
ignore whitespaces in dice sending
fix the test failure which was introduced by a typo
fix profiler frame number stops updating when window is minimized
fix one inconsistency and avoid another issue
style and review fixes fix error code
music add node_type_disc to cmusicdatabasedirectory containssongs
log enqueue block access exception
fix number of threads for pipeline execution
rpc add documentation for deactivating settxfee
changed log message to be more accurate
fix bug where new auths were not findable
update space range
accept on block transactions that dispatch other actions
fix last arc segment co authored by ellensp ellensp hotmail com
apply suggestions from code review co authored by john brownlee mail johnbrownlee com
forgot to advance to next possible name
remove dynamic cast from check
try fix tests
shadowmap fix potentially wrong entitymanager
fix no texture bound warnings in webgl
reverting the last change
fix merge conflict
reformat some code
typechecker remove obsolete check for load expr on typecheckexpr
merge remote tracking branch origin master into master next
fix res trimmed to s on windows
fix regression in row removal
temporarily restore preprocessor checks for c std
minor excluded superflous includes
update diagnosticverifier cpp
dbg finally properly implement setbpactive for hardware breakpoints
gui don t allow inactive breakpoints to be enabled disabled
resolve compile error
resolve type error
resolve format error
resolve else clause formatting error
resolve else clause braces error
improve logging for errors this commit also allows the strings to be translated
add fdbclient knobs initialize unit test
shader memory silence no return value warning silences a warning about control paths not all returning a value
gapi kw fixes remove unreachable code in gisland producer
fix support media shortcuts
fix build error
fix parallel distributed insert select for remote table
init replace url_website with package_url
bugfix rpc add missing commas and correct indentation of explicit elision
bugfix rpc json null is not none
attempt to fix windows compilation error
try fix build
fix lame compile error
try fix tests
add logging for server tracing
try fix summing
remove empty line
fix if exists
fuzz extend descriptor fuzz test
remove duplicated add_child
disable art mutation buffer in redwood until windows build issue is fix
fix videoplayer uniqueid infolabel
add moduledecl base case to extractnearestsourceloc moduledecl is its own declaration context
add two asserts
resolve typo errors
try fix aggregating
try fix aggregating
try fix aggregating
fix build error
fix leak with vulkan allocator
lua fix dialog data for checkboxes if the last pressed button was a checkbox
remove unnecessary panel in features section in the export dialog
shader suld d import storetype
shader image import predcondition
remove disable optimize
first sd status change is silent
silent warning conversion error
add missing linkbutton colors in editor theme
textureregion fix autoslice not created updated properly
save resume data when torrent storage is moved
noisetexture prevent race condition because of ref unref
clang tidy fix
addonversion always store version as lowercase
fix leak with vertex_formats
fix bad include
fix the missing schema field which caused a lot noise in nightly
fixes leaks when running editor
delete commented code
codecompletion don t manually handle equal opeerator
wallet change ismine check in cwallet deladdressbook from assert to failure
qt fix window minimize menu item
unit test for rwlock is set to use timeout for avoiding deadlocks
disable left over centralized healthmonitoring
made clang happy fix build
suppress unused variable warning on macos
fix underlines and striketrough not respecting visible character
lua add possibility to change dialog title with dialog modify title
quote expected strings in parse error messages
lua add dialog color onchange event
update permessagedeflate fuzzing
free items before finishing rendering server
update storagereplicatedmergetree cpp
re enable art mutation buffer in redwood
make traces sevdebug to avoid too verbose
update checkdatapart cpp
update storagebuffer cpp
update storagemerge cpp
fix unit test
merge remote tracking branch origin master into master next
update mergetreedataselectexecutor cpp
fix header inclusion order
preallocate output buffer
make show create table multiline because it more readable and also mysql like
fix signal disconnection soon after connection in editorquickopen
fix regex for s3 uri to pass clang tidy check
fix elu import from onnx
bump format version
bug fix btree header wasn t large enough to hold the largest possible root page references
fix for gcc
tests add coverage of incremental dynamicusage const std set x y s to existing fuzzer
fix multiple clang tidy warnings
windowing wayland gles register using crprendererdma
windowing wayland gles register using cdmaheapbufferobject
windowing wayland gles register using cudmabufferobject
merge remote tracking branch origin master into master next
remove unused ofs variable
fix failing flow flow promisestream callbacks unit test
merge remote tracking branch origin master into master next
update storagereplicatedmergetree cpp
remove unused variable
fix widget previoussibling impl
constraintsystem checking for sk_fix when diagnose ambiguity for generic parameter bindings with no fixes
changed see if we can reuse the language invoker for a menu callback
crprendererdma use texture shader so dma can be used with opengl
windowing wayland gl register using cudmabufferobject
windowing wayland gl register using cdmaheapbufferobject
windowing wayland gl register using crprendererdma
call common function for waiting in rotatebyyawrate make it consistent with other api
fix logged number of inserted rows into replicatedmergetree
vkrasterizer eliminate legacy code
print explanation when fdbcli unlock fails
deleted debug output
addons addons for linux don t necessarily mean they are compatible with freebsd
fix up styler s grudge
typechecker wrapped properties with observers should not have readwriteimplkind modify
silgen remove an unnecessary scope in emitimplicitvalueconstructor
fix a comment
try fix collapsing
move graphite params to separate file
move graphite params to separate file
move graphite params to separate file
allow to rename animation just after it was duplicated show animation name in delete prompt
core test add inputarray matexpr fetch test
diagnosticverifier fix asan issue where twine was stored to a temporary
merge remote tracking branch origin master into master next
merge remote tracking branch origin master into master next
try simplify replacing
try simplify replacing
solved control alt issue
irgen fix simple_display irgendescriptor
irgen type reconstruction supports opaque result types now
serialization implement localdecltableinfo getexternalkey this allows iterating over serialized decl tables which is useful for debugging
guard all of annotateevent with mutex
allow unavailable decls to be walked
tolerate non empty braces in braces
trigger render when scrolling smartredraw
linux drop drm folder from drm includes
merge remote tracking branch origin master into master next
update isconstant cpp
fix wrong whitespaces in debug output
update hashjoin cpp
avoid updating the same endpoint multiple times
buggify master recruit backup worker delay
fix wrong code no changes in behaviour
add documentation for missing infobools
retroplayer crenderbufferdma add sync support
remove old command line option for client
add proper nullptr check
dvdvideocodecdrmprime always use best_effort_timestamp
fastrestore loader send multiple versions of mutations in a message to appliers
update parsedatetimebesteffort cpp
make the assertion in code consistent with the real partition expression
don t remove sni object when sni is lost
fix tests in debug
fixes leak with copytofbshaderrd
fix crash caused by unitialized variable
use openal without direct channels
fix folders visibility above passcode lock
don t scroll down when read from another device
change round return type to float
fix style checker complaint
update mutationsinterpreter cpp
update fdbserver storagecache actor cpp co authored by trevor clinkenbeard trevorclinkenbeard gmail com
update fdbclient managementapi actor cpp co authored by trevor clinkenbeard trevorclinkenbeard gmail com
update fdbserver storageserver actor cpp co authored by trevor clinkenbeard trevorclinkenbeard gmail com
update fdbserver workloads cache actor cpp co authored by trevor clinkenbeard trevorclinkenbeard gmail com
update fdbserver workloads consistencycheck actor cpp co authored by trevor clinkenbeard trevorclinkenbeard gmail com
update msgpack input format
fix clang build tweaked comment
fix logic for skipping broadcast
merge branch mengxu fr send mutations batch pr into mengxu fr code improvement pr
add an assertion that all getrange results are returned in test special key range impl
fix documentation for infolabels that don t support id
fix documentation for infolabels that do support id
align dlie and ngraph average pooling layers
remove stopcondition from mergingsortedblockinputstream
fix compilation issues
merge remote tracking branch origin master into master next
replaced some warnings with infos as perf events measurements are optional
remove unnecessary check for zero determinant in basis orthonormalize
fixes leaks with vulkan device and instance
fix test with vesrioned collapsing
diagnostic non ephemeral pointer init diagnostic should point to init if present
diagnostics missing member diagnostic associated with module should point to name
fix merge related issue
resolve typo errors
merge remote tracking branch origin master into master next
fix style error
update flow network cpp
remove debug output
gui use package_name in exception message
change decaltexture bind from bind_constant to bind_enum_constant
remove more code
fix debug build
fix clang tidy
run springarm3d s process within physics
vulkan populate all fields of vmavulkanfunctions
kernel thread resolve wswitch warnings
reformat the file
fastrestore report error when parsing file has exception
fix wrong condition check
bench remove unused argsman clearargs
decode image fix typo in assert in getcomponentsize
decode shift remove unused variable within shift removes a redundant variable that is already satisfied by the isfull utility function
respond to review comment by rjmccall reinstate fast path where declaration is not a valuedecl
irgen change startoffset to startoffset nfc
merge remote tracking branch origin master into master next
change default upload slot choking limits
fix documentation for container subitem
fix the bool _static logic
pvr tidy value stored but not used
serialization require module files to match on environment too this prevents mixing simulator and non simulator modules
sema remove a fixme
add new line at the end of file
delete extra spaces
lua clear selected colors with app range clear too
decode texture collapse loop down into std generate same behavior less code
decode memory resolve unused variable warning only the first element of the returned pair is ever used
process setupmainthread zero out argument on process start
settings controls only update the displayed value using update after having changed it in onclick
dynarmic enable strict alignment checks also add a missing include
service irs update for new shared memory layout
service time update for new shared memory layout
service hid update for new shared memory layout
core gdbstub updates for new vmm
line unsafe for indexing with known base type unkown identifier
bench replace mempool globabl with test_setup mempool this is a refactor since they are aliases for each other
activate physics process in springarm3d
fix merge issue
simulators only infer simulator environment when there is no environment
remove auto default
fix debug builds
fix presentation error around brackets
buggify fastrestore_get_range_versions_expensive knob for true and false
use default backend in shadowtest
fix ibl in hellopbr
fix crash in strobecolor test
nfc internalize the llvmcontext for llvm ir code generation
nfc internalize the llvmcontext for llvmopt
fix not all remove endpoints were actually remove
backupandparallelrestorecorrectness assert on backup validity
fix compilation of trial version
make logger name colored
service remove unused requestparser instances these aren t used so they should be remove to reduce compilation warnings
service hid npad fix implicit fallthrough errors
drop superfluous locking for atomic in directorymonitor
fastrestore add an assertion in handlerestoresysinforequest as suggested in code review
fix no bed compile error
fix checking if proxy and username pass are empty
fix clearfile if directory passed to it
must close database before returning
fix bad git rebase
avoid unnecessary performtypechecking in repl
fix clang readability container size empty warning in storagedistributed canforcegroupbynomerge
initialize quad_indexed_pass before uint8_pass fixes werror reorder in gcc
merge remote tracking branch origin master into master next
cssimplify fail when try do simplify a key path application to a key path dynamicmemberlookup
include swt_init in log
call the new binaryoperator factory function the constructor is not public any more this fixes the master next build
fix prepare for concatprocessor
gdscript class name existance check enhanced
fix prepare for concatprocessor
nfc fix typo in serializedmoduleloader cpp
audren lower log level for requestupdateimpl to debug
npad lower log level for vibratecontroller to debug
fix crash disabling extensions
curlfile proper setup proxy credentials
fix inadvertent savedversion update
backup worker pops max of savedversion or noop s popversion
use proper depth buffer format for rgba shadows
android ignore intents until application is initialized
update altercommands cpp
update altercommands cpp
move the callbacks into run right before it exits stopped true doesn t cause the run loop to immediately exit
cleanup use the shutdown callback for destroying tls state
loader elf fix process initialization using programmetadata default
opt eliminate whilecontrolstmt with non zero const conditions
fix comments on startversion in backupworker
cs fix an if statement we should be printing retired constraints if we have any
another stringref std string conversion fix this is needed to make this code compile on master next
fix a test failure minrestorableversion may be decided by snapshot version and is larger than contiguouslogend version
adjust cursor blinking effect for on screen keyboard to accommodate arabic text
add opus and oga files as song formats
fix constantexpressiontemplate with arrays
scroll history to bottom on sending a message
improve poll closing by timer and results reloading
emit signal when animation ends by seek
merge remote tracking branch origin master into master next
change simulation storage server selection back to random
fix local button not being pressed in the scene dock on first run
settings don t load setting values twice
use python style formatting in clog setloglevel
fix log levels in debug builds
register the displayserver singleton
skip redundant file dir tests
call first_load from manage_media
use nullptr instead of null
nfc indicate foreign modules in ast dumps
network fix overflow in our network code
fix mac os build
async gpu only do reactive flushing on extreme level
maxwelldma correct copying on accuracy level
not remove metadata directory when attach database fails
update fdbserver datadistribution actor cpp
check that only embedded timezones are used in debug build
update databasefactory cpp
tests add fuzzing harness for functions in primitives block h
fix compile errors
drop unused memorytrackingformerges from context it set in src storages mergetree mergelist cpp anyway
fix camera mode optstr
mono c fix assemblies being reloaded a second time unnecesarily
boop the ci
fix format error in performance statistics formatting
fix core tests
exit fdbserver after a receivererror because a packet which should be delivered will never be received
constraintsystem fix new empty locator in resolvevaluemember
fix crash when torrent is deleted on limit reached
pvr use active channel name when importing guide from clients
flowtransport increase delay for connecting to failed connections
mono c fix load hook not called for some assemblies on domain reload
create legacyscriptpubkeyman when not a descriptor wallet
return nullptr from getlegacyscriptpubkeyman if descriptor wallet
correctly check for default wallet
generate new descriptors when encrypting
implement getmetadata in descriptorscriptpubkeyman
implement signmessage for descriptor wallets
implement returndestination in descriptorscriptpubkeyman
implement getreserveddestination in descriptorscriptpubkeyman
implement getid for descriptorscriptpubkeyman
implement ishdenabled in descriptorscriptpubkeyman
implement markunusedaddresses in descriptorscriptpubkeyman
fix double params
remove brittle assert the vtables are sorted by address so aslr makes this test non deterministic
patch tool_change typo
svc remove unused variable since the vmm refactor this is no longer used or needed
update databasecatalog cpp
fix crash when changing time value of multiple animation keys at once via inspector
fix memory leak in load_document
merge remote tracking branch origin master into master next
serialization don t allow simulator and non simulator modules to mix
merge remote tracking branch origin master into master next
tried to fix build fail fix codestyle
remove obsolete comment
addons pvr remove not used lambda for this on getdrivespace call
fix issue showing the dynamics popup when the ui scaling changed
vulkan fix invalidimagelayout validation errors it is illegal for shaders to sample from images that have a blit layout
restore dynamicsfield selected status when popup is closed
merge remote tracking branch origin master into master next
add buggify back to desired_teams_per_server knob kudos to a j who found this bug
dependency scanner get the clang module file name from clang
dependency scanning cope with inferred module maps
extui allow small jd mm
avoid processing all batches before distributed shutdown
fix visible drift of memorytracking metric
remove squashingblockoutputstream from the stream to allow blocks from watch query to propagate to target table as is
subtitles fix smi subtitles to use quoted start tags
addons do not submit install jobs in the background
fix odbc bridge test
cleanup on exit
better fix for texture unit spew
fastrestore loader use shallow copy for single key mutations when send mutation messages to appliers
irowinputformat performance fix
fastrestore loader free memory after send
fix unwarm regdata index
pvr get previous epg tag instead of previous of previous
fix bug where spinbox would not update to it s actual value after non numeric input
qt remove todo bug fix for old versions of qt
fix typo in optionsdialog cpp qbittorent qbittorrent
tests add fuzzing coverage for jsonrpctransactionerror and rpcerrorfromtransactionerror
tests add fuzzing coverage for parsefixedpoint
tests add fuzzing coverage for formathdkeypath and writehdkeypath
tests add fuzzing coverage for check_nonfatal
tests add fuzzing coverage for stringforfeereason
make line2d be white by default
wallet recommned absolute path for dumpwallet
spacing co authored by christian clauss cclauss me com
lcd preheat followup
strange fix how it was possible earlier
fix in istorage
fix after refactoring
fix style check
use the updated field name in status when fetching process busyness info
dbg remove a check that has been proven stable in memoryreadsafepage
fix buffer stuff
patch xy frequency menu
support write for xdbcstorage
fix schema of storage
stepper_timer_rate might call a function
suppress compile warnings
restore setfamily in crash report window since setfamily in qapplication setfont was remove
updated comment to reflect listqualifiedusers better
fix copy paste array index bug
don t override format value
fix select of column alias which default expression type different from column type
drop unused client_version_suffix macro
remove unneccessary increment in textedit
fix backupworkerenabled flag not set bug for first backup this can cause restore failures because the latestlogendversion could be wrong
fix the interface of special key space only keys in xff xff xff xff xff are valid
update special key space argument
rpc do not advertise dumptxoutset as a way to flush the chainstate
fastrestore fix windows build
remove redundant std
temp test appveoyr
fix scaling in crash reporter
net processing ignore unknown inv types in getdata messages co authored by john newbery john johnnewbery com
different approach to proper colouring of filters
fix assert that can fail when loading a sequence of files
fix compile error
some knob tweaks
docs improve commenting in processgetdata
revert writeduringread change
assert tr options checkwritesenabled
fix bug emit kv pair if we end in a conflict range
tbdgen fix unused variable warning
silgen eliminate stale comment the verifier is done this is the correct thing to do qed
merge remote tracking branch origin master into master next
test fix message for ecc_initsanitycheck test openssl is long gone
enable dynamics for eraser
add sampling memory profiler
flow platform cpp
flow look for plugins in the platform specific place
flow use cpuset
network use application name for web server auth prompt
fix fdbrestore parameters with one leading hyphen instead of two which did not match documentation
drop extra marduino includes
fix shard numbers output in logs full cluster had been printed over optimized
fix disable sanity check backup metadata file which can increase the false positive rate of toomanyfiles error
re add improperly named options in order to not break compatibility with anything depending on them in a patch release
fix macos compilation error
fix eeprom init glitch
onerror wasn t properly retrying tag throttled errors
c header updates
update shortcuts cpp
fix resetting menu scroll after refresh of filter list
fix crash when user pins chat in remotely remove filter
remove info display for last messages in saved messages
fix crash in theme editor
remove finished todo comment
update makecontinuousmemoryfromcheckpointtopos in peekablereadbuffer
remove double space
sema remove some dead code the parser should ensure that we never encounter this case
diagnostics fix missingconformancefailure diagnoseaserror to use solution to retrieve fixes
diagnostics fix contextualfailure to use solution data for hasappliedself
add unit test
update mergetreeindices cpp
diagnostics use solution for verify presence of fix in diagnoseuseofreferenceequalityoperator
updated dbinfo broadcast to minimize the number of possible wan round trips
test setversionstampedkey without ryw enabled
append newline to printout of defaulted constraints
atomicopsworkload lower severity for txncommitunknownresult
fix stale ibl string in shadowtest sample
fix unused variable warning in atomicparallelrestore
move delayedhealthupdatef declaration
fix decoration applying and trigger repainting on update for some reason this is needed for newer qt in flatpak
remove microsoft comment regarding dirent in parsingchecks
commit some lines that were intended to be part of earlier commit
fix bug where readversion member future wasn t assigned
check if client is newer than server
upnp prefer movie posters to thumbs when serving the library items over upnp
update arrowblockinputformat cpp
fix bad code
clean up m122 code
fix motor current array
update threadstatusext cpp
fix performance issue in update_bitmask_region fallback
mention offending function name in indented block expected error
remove unused error codes
add missed include
csdiagnostics break to the fallbacks when no specific diagnostics is found
register renderingdevice as gdscript singleton
ast assert we were given a type parameter in genericsignatureimpl requiresclass
decrease chance of false positive in random generator testing
increate total read length while reading
correct stop condition
update compressionfactory cpp
fix bug with dictisin
remove views count from admin log
pass stream_cancelled to the copydata as stub in storagekafka
force destrying of kafka buffers from the shutdown in storagekafka
remove extra check
better exception message
add read lock for correctness when determining whether a rollback is required consider moving this to an atomic high watermark
fix double space
add a separate string for empty channel admin log
fix mutations assignment
fix limit for mergingsorted with fast forward optimization
fix compiler errors
init all input ports in groupingaggregatedtransform
init all input ports in groupingaggregatedtransform
remove headers vscode auto add
wrap some macros
rpc createwallet warning that descriptor wallets are experimental
fix addwellknownendpoint order bug
pvr cpvrepgdatabase replace c style casts with c style casts
go a different route
event server keyboard events dont work for keys that have long press defined
assign ctrlm as default shortcut for invoking mark dialog
correct suppression event
more aggressively cleanup a bad process id file in simulation
fix use of siltypesubstituter and typesubstitutionfn
remove createeliminatedeadmemberspass which breaks ubo layout this fixes a crasher when using optimize size
add batchindex in loading files
rpc blockchain cpp remove g_utxosetscan mutex that is only protecting a single atomic variable
claim trailing closure arguments like we used to in error cases
parse report success if is found for multiple trailing closures
merge remote tracking branch origin master into master next
bind set_suffix get_suffix in tree
nfc sema collapse invalid where clause diagnosis
add check in storageimplinforequest evaluate instead
try hashmap from common lib
fix for new use of callexpr create
compilation error fix
fix rebase issue
implement missing destructor
loose some limitation
fix the compilation error
qt display warnings as rich text
qt reset toolbar after all wallets are closed
silownershipverifier fix a compiler error in the no assert build
ffmpeg version in matrix returns hls not hls applehttp for hls format
merge remote tracking branch origin master into master next
remove wintab error message
add support for alter rename column query to distributed table engine
fix compilation error
simulation did not properly track exclusions of tls processes
fix invalid trace event detail named type
metrics add table settings block
fix to facilitate branch merges
fix merge error
remove unneeded module violation
rename type transactiontype
rotation gizmo visible false when camera preview
don t set special_key_space_relaxed for test within one module
refactor remove override for final overriders
skip channels without write access in forward box
improved a comment
fuzz fix vector size problem in system fuzzer co authored by pieter wuille pieter wuille gmail com
fix descriptive comment
shared build fix
save user data changes correctly in current recovery session
remove a wrong assert
fix crashes running tests from cli
control trace interval with a knob
only call getkeyrange once
apply clang format
dont send referer on redirects same behaviour as ffmpeg ans browsers
merge remote tracking branch origin master into master next
fix for fanartres not being used in advancedsettings xml
try fix tests
try fix tests
try nut update scalars for only_analyze
create local context for query analyzing in pushingtoviewsblockoutputstream
update randomstring cpp
update randomstring cpp
microopt stream single char instead of single char string
fix merge conflict
update datatypecustomgeo cpp
vkpipelinecache use a null shader on invalid address
revert strage class monitor leader because of correctness issues
asserts in datelutimpl
fix race condition in replicatedmergetreequeue
range with non numeric const argument crash fix
tempted by the const of a seg_length
remove default value from overriden functions in ireservation children
prevent capsuleshape2d height from being less than zero
tweak the editor log selection color to match the current editor theme this overrides the default blue color
implement csharpscript inherits_script
use src_type for convertion in keycondition
add a comment
add a comment
fix the first cpu estimate could be wrong enough to cause stalls on a newly recruited proxy
fix ubsan and msan report in datelut
fix msan report in mergetreeindexfulltext
remove new line
remove obsolete suppression
using command comma on macos as default shortcut for editor settings
only check if items are iso and expand disc image when in file view
fix memory leak with light maps
fix typos and whitespace
update functionfactory cpp
optimize block sortcolumns suggested by alexey milovidov
fix style check in block sortcolumns sigh
remove unnecessary call to getuserpercentage in application cpp
better support for sanitizers
tweak jd jerk limit
fix call window hiding when compositing is not supported
remove replyto from switchinlinebotbutton in same peer
remove unnecessary include
frontend remove tracking for context wrapper
remove strcpy compile warning
disable create polls in support accounts
silence eq band c1 c2 and c3 may be used uninitialized warnings
try fix merge with master
don t reset search results on dialogs re open
add exception for not supporting selective load to direct dictionary
destroy box before potentially exiting function
memo fix example
try fix sample final
clangimporter ignore matrix types for now
adjust to new elementcount type for vector operations
irgen adjust to add matrix type ignore for now
changed formatting of format cpp to fit clickhouse s
update format cpp
alt backspace deletes a word too on ui entry fields
fix invalid memory access in readhotdetection test due to memory being stored in a non standalone stringref
application fix mixed pictures videos slideshow ending after first video
fix ratekeeper was logging the auto throttle count for the manual throttle count
fix busiestreadtag was logged before the first interval
fix bad merge
regression var declaration type info parser bug fix
fastrestore check restore request must have been set when master needs it
move comment about swap meaning for deserialize to correct function
disable checkbox if torrent doesn t have root folder
fix c bindings after lightmapper changes
fixes previous changes
another std string stringref assignment
allow godot to kill it s own pid
set parser error when infer type is null
perviewuib header and cpp padding didn t match it wouldn t really cause any issues because still
silgen assert the sourcefile has been type checked
changing the last suggestions from review
eliminate undefined behaviour
fix signature of hidden actionwrapper assignment operator
fix windows build
remove polygon border checks
sil fix a use list iteration problem in replacephiargumentandreplacealluses
silprinter be more tolerant if an instruction is not contained in a parent block
deadobjectelimination reuse code nfc
android fix crash on kodi exit powersyscall no more available but called
log button name in event client
update mongodbdictionarysource cpp
update mongodbdictionarysource cpp
try fix tests
tests fill fuzzing coverage gaps for functions in primitives block h
delete extra lines
onnx sub is binaryop
fuse rsub rdiv with scalar
silence interp resize_type warning during ncnnoptimize
fix renaming rss autodownload rule
fix build without opencl
fix log tables
fix log tables
silence mulitple may be used uninitialized warnings in renderingdevicevulkan uniform_set_create
fix config prefix
do not use overloaded comma operator
accidental override on non inherited function which strangely only broke the windows build
increase backup worker memory buffer size in simulation lower limit causes backup worker to become stuck
nfc initassociatedtypewitness is now unused in non ptrauth builds
try fix tests
merge remote tracking branch origin master into master next
try simplify code
flush to zero convolution denormal weights
don t delete extra char using ctrl backspace to delete words in ui entry
add an assert in case of write into a file after close
narrow keyrange of special key space to xff xff xff xff xff initilize new ones in correctness tests
update special key space comments
add an assertion in singlespecialkeyimpl
update error tracking in read write conflict range
using coinflip to simplify the code
build style fix
try fix build
priorbox test fix
try fix tests
fix popup window size calculation in popup_centered_ratio by using rect2
never throw from destructor
try to fix use after free error in mergetree
update datatypesdecimal cpp
remove unused header file
addons fix open of addon settings dialog via addon there was wrongly the missing and always has broken the open
fix symlinking data directory for atomic database
updated comment for the condition where a transaction relay is denied
log remove no rpcpassword set from logs
revert try to fix use after free error in mergetree
comment a motion function
g29 patch followup
prevent album artists being left blank when merging scraped album results and preferring scraper results enabled
nfc ast push up a fast path in typebase getcontextsubstitutions
fix style check
ignore tests dependent on flann when flann module is absent
check if item is playable before add play from here or play only this to context menu
remove unnecessary is_const_ref struct
try fix tests
remove some superfluous includes
add nchar and nvarchar synonims
update httphandlerfactory cpp
delete unnecessary comments
update the permission string for the oculus hand tracking to match the latest api update
fix current counter id check
sil use limit ondemand for an enum element in sildeclref getlinkage
refactor remove unused bool parameter in blocknotifygenesiswait
always specify content type in web api error responses
fix too eager gdscriptfunctionstate stack cleanup
try fix compatibility
acutally use the variable i introduced
pvr fix context menu item add timer visibility conditions
change docs for walletcreatefundedpsbt rpc method
metrics bug fix page build count was being updated in an incorrect place
fix an infinite loop
parse events through the eventparser
gardening remove addressed fixme
merge remote tracking branch origin master into master next
fix indefinitely bouncing of dock icon
add esc shortcut to clear selection in section of scheduled messages
fix album items selection in section of scheduled messages this bug relates only to albums with captions
update storagesystemparts cpp
expose image save_png_to_buffer method
sourcekit don t handle operationkind all
remove erroneous restore options
backup s programstart command line string would be incomplete if arguments were specified using
fix code style issues
remove useless comment
another compiler fix reduced the size of the endpointnotfound map
fix include paths
add a comment on restore s clear mutation
fix arrow stream termination
fix issues with slider focus and scroll input
test set logthreadnames in unit tests
tweak the import compression property hint for clarity
less verbose logging in mutation finalization task
fix segmentation fault when using context_vulkan after memdelete context_vulkan
fix dynamically allocate all arrays
logging perf event settings name when event is not supported
more precise number of total events
property wrappers for properties that are implicitly default initialized via property wrapper mark the init call as implicit
runtime repair the android build the android log symbol is print rather than printf
fix bug in which the wrong box was used
wrapped all the functions inside a class
wrapped all the functions inside a class
fix system tables output
doc add and fix comments about never destroyed objects
define fill_toolloop_params_from_tool_preferences only when needed
add missing include in engine cpp
check original title during movie search
fix codetexteditor not respecting focus in _input
auto completion for throttle disable auto were wrong because of logical operator grouping problems
merge remote tracking branch origin master into master next
pvr fix epg tag path not initialzed when loading the tag from database
fix clang tidy
prespecialized metadata allow existential arguments
drop check for getpositionbyname it already has it in expressionactions
dvdvideocodecdrmprime enable use of threads
make gcc happy
another compile fix
ast don t clone typereprs in paramdecl clonewithouttype
sema don t clone typereprs in constraintsystem shrink
comments and clarifications
comments and clarifications
add a comment
increase the default directionallight3d and omnilight3d shadow biases this should decrease the amount of visible shadow acne by default
apply all transformations again
apply all transformations again
fix file dialog filename cleared when selecting favorites
gbm add virtualbox
pvr fix setting initial recording size value from addon
fix test looks like it was incorrect
fix clang tidy
csdiagnostics adjust tuple contextual mismatch diagnostics to handle subscript assign to source
fix switched params
fix param path bug
marlin use strcmp_p
don t re reverse sort
disable fallback session management
add samsung exynos to drmutils cpp
fix js audiocontext parameters were not passed along correctly latencyhint is supposed to be in seconds not milliseconds
doc update unbroadcast description in rpc results
doc provide rationale for randomization in scheduling
mempool don t throw expected error message when upgrading
skip extra newline in tscn when renaming dependency
fix build after merge
colorpicker margin bug fix
don t spam logs if there are no dbus
fix pipelineexecutor executestepimpl
fix pipelineexecutor executestepimpl
fix primary key check
switch ngraph op v1 multiply to v0
fix comparisons between presumed and buffer line numbers
gardening remove dead code in serializedoc cpp
index apply index ignore system modules to clang modules
ignore invalid tablet driver name when non are available
fix initialising song count later use to reserve json results
qt increase console command max length
fix bad merge
gdscript fix assert message when no custom message is set
wallet make periodicflush uninterruptible
fix context usage
use compile time constant for the size of the signal parameters
enable raycast nodes by default
fix style errors
fix style errors
make single if
modified comment block
further improved comment blocks
return raw scores in confidence value
destroy descriptorsets properly
fix windows compile bug
free dynamically allocated memory
fix style check for popcnt check
walletdb remove unsed boost thread
android call codec release
nfc remove extra deduplication of external depends clang s dependencycollector already deduplicates entries by the provided file path
merge remote tracking branch origin master into master next
remove wrong comment
lua fail if we give an invalid parent to layer parent setter
make copying directory string more concise
merge remote tracking branch origin master into master next
add prefix facility to librdkafka logs
lua fix crash when setting layer parent to nil
remove dead code
clangimporter add explicit std string conversion
try fix tests
cowardly refusing to write and empty batch
don t spam logs with missing keys
fix style issue
try fix copier tests
try fix tests
vp fix short stream changes
do not probe joypads if directinput cannot be initialized
try fix tests
actually set gdscript static reference
fix compiler error
try fix tests
fix bug with throttler and query speed estimation
update fdbserver logrouter actor cpp co authored by a j beamon ajbeamon users noreply github com
fix bad formatting in clickhouse client
minor fix to code style
move sending crash reports below logging
add debug output for vtable layout
emit a warning if server was build in debug or with sanitizers
gardening fix a double the in a comment
constraintsystem add missing handling of stmt to getloc
minor cleanup of postprocessmanager ctor make sure all fields are always initialized and the ide stops complaining
ship s onblock test should consider trxs with one or more actions
fixes building mono release templates
merge remote tracking branch origin master into master next
fix arcadia build
infolabels expose player progress and player progresscache as a string
add empty line
update osx guid to sdl uid conversion co authored by john wakley johncwakley users noreply github com
fix cascadedetect convert from old cascade to new
update pullingpipelineexecutor cpp
fix very rare race condition in threadpool
update trap cpp
apply misc tern
android remove video_scaling_mode_scale_to_fit_with_cropping
update mergetreedatamergermutator cpp
remove trailing whitespaces from formatted queries in some cases
fix build with old vulkan sdk
python add delayedcallguard to xbmcgui dialog info
fix cpu only build
report dictionary name on dictionary load errors
update topictree fuzzing
fix crash in still downloaded documentdata
remove debug inline bot results marks
fix stickers panel on retina screens
allow export of just converted supergroup
fix a crash in custom notifications
add cinnamon sound settings command
rename sandboxautostart to portalautostart
fix crash in gtk file dialog on wayland
remove duplicate log line
fix little typo in theme names generator
fix crash in context menu for uploading scheduled messages
fix build for macos
fix sanitizers build
try fix build
try fix build
better log messages in configreloader
adds crisp shape rendering for svg file format
remove debug output
fix unit test with ubsan
update pushingtoviewsblockoutputstream cpp
update pushingtoviewsblockoutputstream cpp
fix nullptr dereference in system query and lazy database
fix compiling when disabled
cli create getnewaddress
cli create generatetoaddressrequesthandler class
cli add multiwallet capability to getnewaddress and generate
fix compiling when disabled
remove structure lock on parts send
fix build on windows
disable filtration for partialsortingtransform
fix for expression class inner string can be single quoted
clase all inputs in sortingtransform
wallet remove unused boost thread_interrupted findwallettx is only called by zapwallet which is never called in a boost thread
fix parseredismessage use after free
fix build of test
add new enum value clang builtintype incompletematrixidx to switch statements
address review comments
macos build fix
fix minor inconsistency with master branch
further reverse iteration work
don t re use cursors
fix crash in load_model when gpu is not used
never use custom code for portal detecting in flatpak
fix visual glitch in filter change from archived chat
please build already
let s be optimistic it will build eventually
change ok text in snap dialog to ok
fixes the right and center alignment bug of rich text label
remove debug print
add checks for empty server name
vulkan implement inserteventmarker note that pushgroup popgroup are already supported
made property hint consistent with other property hints
docs add missing backtick to option description it broke the startup option table generation
add a warning if attempting to use rocksdb without it built
fix reverse iteration seek behavior
no longer use a tailing iterator
predefined var check for for loop counter
ast remove some pointless uses of hassingleexpressionbody
skip some gapi tests if videocapture is not capable to playback video
add comment to m281
fix crash in streaming loading of a document
fix crash in saving of a document
update src interpreters databasecatalog cpp
update src interpreters databasecatalog cpp
check if the specified android release keystore exists
don t use deprecated qprocess startdetached overload
don t use deprecated qurl topleveldomain
don t use deprecated qbuttongroup buttonclicked
don t use deprecated qabstractsocket error
downgrade handle not signaled error to trace clogs logs quite a bit
enable text selection in execute output of editornode
txindex remove unused boost thread
dbg allow setting temporary labels from the labelset command
bridge allow optional parameters for dbgloopget
dbg truncate labels in labelget
dbg improve working set mitigation
dbg properly truncate the page info
uppercase registers in trace view
fix a bug when updating disassembler config in trace viewer
keep highlighted token highlighted
refresh rocksdb iterators instead of allocating new ones
fix readability qualified auto
inputstream cleanup extra whitespace
also check for users xml elements in config xml
net reformat excessively long if condition into multiple lines can be reviewed with the git option word diff regex
remove debug output
remove debug output
update compressioncodecdoubledelta cpp
update compressioncodecgorilla cpp
fix ctrl click not working for subclasses
revert fix parseredismessage
fix possible crash in editcaptionbox
make clang tidy happy
merge branch svg rendering
clarify a comment
add debug output when reboot ids change
generaterandom check that all arguments are literals
refactor replace recursivemutex with mutex in shutdown
ignore throttling errors in fuzz tester
fix same error in values as well
restore capture caches when missing
disable columnmatch columnname logic
restoreloader wait on actorcollection error so that we will not fail sildently
fastrestore sanity check actors do not throw error silently
android fix stall on stream end
fix undefined behaviour of doubledelta
refactor replace recursivemutex with mutex in timedata cpp
fix editor crash when mbedtls is disabled
update storagedistributed cpp
minor fix on glsl shader parser
simplest fix to a dependency bug better commenting out
fix build checked locally no pr
pvr margin change margin values should be visible for epg based timers
update dnsresolver cpp
servicemanager init pvr manager after login not already on login screen
android fix drain in flushed state
track upstream changes
fix fbx import assimp error
inputstream log error for use of deprecated inputstreamaddon property
log remove deprecated db log category
update dnsresolver cpp
typecheckconstraints check for archetypes when warning about unrelated type conversion
autocompletion support for users in client
fix rest errors found by style check
fix crash in media viewer refresh in showphoto
rpc getaddressinfo rpcresult fixup
remove an invalid assert
fix gcc build
fix topictree fuzz build
remove log debug
remove redundant error code
refactor add aborterror alias
fix benign race condition during shutdown
restoreapplier remove unnecessary txn reset
video re add mark watched unwatched context menu items to content less video sources
remove call in watch actor instead of transaction watch
call sha256autodetect in benchmark setup
revert autocompletion support for users in client
runtime clean up verifymanglednameroundtrip comment
add g code backspace support
restoreapplier remove getvalue actor
dnn ngraph add nullptr checks
old qcom driver cannot handle binding id alias
allow original release date for albums to be scraped
merge remote tracking branch origin master into master next
fix under msan
fix manual album refresh
fix comments bug
nfc clarify comment on metatype casts on typecheckconstraints
fix crash when creating new text file with no name
use hevc_nal_cra_nut as keyframe
merge remote tracking branch origin master into master next
fix scene tree showing up when the root selection is present
compile error fix
kernel account for system resource size for memory usage gettotalphysicalmemoryavailablewithoutsystemresource gettotalphysicalmemoryusedwithoutsystemresource seem to subtract the resource size from the usage
remove unused local variables
update http fuzzing
address review feedback
fix compile error in scandependencies
don t miss columns ttls update
update parsershowtablesquery cpp
fix class memaccess warnings
remove weird include add by clangd
crash when trying to set up the rocks engine without building rocks
remove another wrong include add by clangd
yuzu configuration show assembly shaders check box
fix windows build failure
remove blank tab
add missing header
remove extra columns from sample in storagebuffer
fix address formatting
fix corner case only for exit code overflow
update storagereplicatedmergetree cpp
s null nullptr co authored by a j beamon ajbeamon users noreply github com
no longer double delete the db
fix nullable prewhere column type
refactor remove unused request fhelp
update client cpp
update client cpp
nfc consistently return errortypes when resolvetype fails
nfc remove typechecker getsubstringtype inline it into its only caller
nfc fixup resolvesilfunctiontype to not return type
fix compilation with g
simplify some code in platform uwp export
typechecktype avoid diagnose convention representation when function type has error type
fix fdbbackup help message
tell openldap too create a new ssl tls context for each connection
delete on dealloc
try fix tests
print progress bar even if there is only one file
constraintsystem nfc leave notes on the algorithm used by diagnoseambiguitywithfixes
constraintsystem avoid recording duplicate fixes for contextual type mismatch
constraintsystem nfc debug log solutions before attempting to diagnose ambiguity with fixes
fixes the stopping of animation effects in bbcode after appending
typechecktype lift malformed type check for fntype in resolveastfunctiontype
fix arcadia build
fix favourite project button in project manager
ensure pair callback data is set to null when it s null
pvr make cguiepggridcontainer updatelayout thread safe some of the modified members will be read by timeline refresh thread
re enable scroll follow on richtextlabel clear
fix faulty thumbnail hashing
make assert strings consistent
clang format again
welch t test
fix calculating full names of row policies
fix parsing create settings profile with writable keyword
fix formatting of create dictionary
merge remote tracking branch origin master into master next
rpc rephrase generatetoaddress help and use package_name
net add seed bitcoin wiz biz to dns seeds
remove plugin dispatcher
fix copy paste error
don t log ok status on db close
fix whitespace issue in sum_of_digits
welch t test
welch t test
use path instead classname to prevent errors for exported visual shaders
don t rewrite query in client when there are no query parameters
fix windows cr lf detection in libmarkup
merge remote tracking branch origin master into master next
welch t test
remove unused variables these currently warn for being unused
fix clang static analyzer
avoid connection to replica when fetches are cancelled
docs comment space fix
remove extra vertical space
do not materialize header for merge tables
update assert string
add source steps
fix sending image from clipboard as file
fix wrong z order of drag areas that were under schedule button
fix default path to use to save a recovered document
strokewidth remove unused local variable fixes compiler warning signed off by stefan weil sw weilnetz de
fix race condition
fix core babystep compile
remove outdated comment
fix framebuffer size on fractional scaling display
simplify transform2d get_rotation
fix getreadversionreply recentrequests
remove extra line
add small comment
missed change for primary key
update partialsortingtransform cpp
fix segmentation fault
add rollupstep and cubestep
driver tbd files are linker inputs
fix dwin typo
remove repeated headers
one more test for read through boundary
update lib ast diagnosticengine cpp co authored by suyash srijan suyashsrijan outlook com
cancel fetches before metadata alter
codecompletion update codecompletionexpr dump to dump base exprssion
return boxed long when marshalling a godot int to mono runtime in dynamic contexts
merge remote tracking branch origin master into master next
add const to rocksdbkeyvaluestore methods to support new ikeyvaluestore interface
fix formatting code style
fix getwriteversion and getlastcommittedversion signatures
adapt a swift master change to master next stringref vs std string
fix storage buffer metadata
move sha256hash to its original position it s not needed to have it in its previous position anymore
frontend move parsing outputs to end of pipeline
frontend move generic sig verification to end of pipeline
remove unused method better
use seekforprev to simplify reverse iteration logic
re use rocksdb iterators for scans
increase read concurrency
sim2 remove an useless line
silently reject empty passwords empty user names are rejected verbosely
fix querypipeline unitepipelines
fastrestore test generate random addprefix
fix for windows
fastrestore evaluate addprefix test is used
add check for syskeydown on control and alt
fastrestore fix testing on addprefix
fastrestore transform keys when db is locked
hide notification position in settings on wayland
sil opt add enableprettystacktraceonsiginfoforthisthread like we do in the frontend
android fix drain in flushed state
emit signal when redirected to magnet
don t modify source url when wait for download status
limit the number of threads
update interpreterselectquery cpp
disabling dafault nms in yolo layer
add missed check
refactor block import implies isinitialblockdownload
fix context leak in interpreterselectquery
change string type to binary
merge branch master into master
try fix totals
remove an apparently unnecessary conversion cscript cscriptid scripthash is unnecessary because scripthash and cscriptid do the same thing
remove unused variable
merge remote tracking branch origin master into master next
add translation of current boxcat events
avoid reallocating for each batch in commitbatcher
fix distinct for finishsortingstep
print build id on startup
fix sleep in signal handler
turn off traps
update image and details section of documentation
make epsilon equal system epsilon
set author after program details
remove dependency on tzdata
fix mysql syntax for trigger tgrinsertgenre
update kafkablockinputstream cpp
update allocator cpp
fix false msan report in symbolindex
albumscraper allow fine grained relevance values
update mockedbroadcastingechoserver cpp
make constants in cpp file extern
remove unnecessary conversion
remove redundant header includes
add limits header file
one loop call to ui update
rpc simplify getalloutputtypes with the join helper this commit does not change behavior
fix insecure call of tprintf signed off by stefan weil sw weilnetz de
ensure search window is increasing and greater than tolerance
add cap arcs
clean up setup
add see also reference to brents method
fix for sleep using stl
fix doc for define
quickie g35 patch
conditional compilation for network reader
core logger complete initialization of logger structures for using of logging functions from global destructors
bug fix accidental local variable redeclaration
test add two edge case tests for csubnet
reorder variables to comply with the auzure build pipeline
update flow platform actor cpp co authored by a j beamon ajbeamon users noreply github com
fix mistakenly swapped previous and current lock orders
fix header for nullable prewhere column
optimize mish for cpu backend
remove a test so that it is all ascii
depscan nfc remove check for ononesupport this dependency is already recorded in the importinfo s modulenames
changed default tls handshake thread count
these compilers are really pedantic
merge remote tracking branch origin master into master next
binary search tree insert function modify insert the greater value to right smaller value to left
small quality of life indication that mapped button is gc
lm silence no return value warning
backupandparallelrestorecorrectnessworkload change unnecessary key to keyref
style fix bump tests
fix add missing header
fix incorrect constructor delegation
fix arrow operator instead of dot
fix function return
fix lint errors
docs copyright message
fix add missing header for std time
fix fix forgotten namespace
disable restart on session start explicitly
fix ability to open more than one calendar in schedule box
windows fix for dual monitor problem
be sure to set correct pip maximum size
add missed handler for mtpupdates_getstate
fix launching with autoupdates disabled
fix invalid cache settings being read
fix crash in session destruction
fix create group box
feat add polar form initialisation to our complex class
fix cpplint issues
fix cpplint issues
fix readability issues
fix line spacing
fix line spacing
bug fix heap sort fresh implementation
fix line spacing in tests
add author name
fix compiler error
fix cpplint long int64
multiplication result may overflow int before it is converted to long
code quality fix deallocate resorces
update storagesystemtables cpp
fix style check remove unused logical_error
fastrestore test both addprefix and not add it
improve debug message in colpartition computelimits signed off by stefan weil sw weilnetz de
merge remote tracking branch origin master into master next
fix release allocated memory
fix convert global variables to local
fix use delete operator
fix use define
fix wrong function name
remove unused imports
fix logging output
remove unused code
slightly modernize code around zookeeper
qt remove needless headers
fix unitialized memory
prevent nullptr dereference on swkbd error case
docs typo fix
feat add exit option
merge remote tracking branch origin master into master next
remove unneeded option
fix minor docs issue
fix material preset type
fix spaces between include and header file co authored by david leal halfpacho gmail com
fastrestore fix avoid reset txn backoff time in sanity check
fix null point bug because omit calling createinstance
fix compile warning
refactor use mutex type for g_cs_recent_confirmed_transactions
wrapped example in pre
fix saving of main menu accounts state
fix repeated sentences
fix online status stuck when switching between accounts
add wiki link in file brieff
improve detected inconsistent lock order error message
apply suggestions from code review co authored by david leal halfpacho gmail com
add cannot_parse_uuid extern const
fix another dangling reference problem
atomicrestore fix typo in comments
update asynchronousmetrics cpp
tests provide main function in fuzzer
remove unnecessary newline
clang format again
use std filesystem path
fix intermittent invalidimagelayout error for main depth buffer
fix typo in dwin preheat array
fix migrating background image
fix adjustable colors in default theme
fix minor typo
fix protocol check
fix closing float player over intro
include cstdlib for std abs
replace cstdlib with cmath for float overload of std abs
fix style check
use better test function names to avoid conflict
update function name again
rpc check that left section is not multiline
fix crash in filters update
btm give better names for unknown functions
oops fix typo
fix bad log message at server startup
add a comment with some thoughts
create query fix for replicated dbs
better diagnostics of replica appears to be already active message
fix recursive propose for drop database db replicated query
refactor some code
prevents usage of return in main shader functions
yuzucmd tester correct execution
svc remove global hle lock
nvdrv remove frame limiting as host timing already takes care
scheduler correct assert
scheduler fix hle threads on guard
scheduler correct locking for hle threads
synchronization correct wide assertion
framelimiting enable frame limiting for single core
singlecore correct ticks reset to be on preemption
coretiming cycletimer correct idling
dynarmic interface don t clear cache if jit has not been created
yuzuqt hide speed ui on multicore
qt remove unused includes
generate inspector preview for image resources
fastrestore loader frees mutation memory immediately after send mutations
restoreloader clear memory per loading param
test perf without gracefully clear memory
videostreamgdnative close file in cleanup we should close the file handle when we are done
really ignore the error from uploading actor
fuzz with dedicated compressor
am stub getindirectlayerconsumerhandle needed by monster hunter generations ultimate
addons fix definition of isuserinstalled addons
update errorcodes cpp
suppress unused var warning
fix clang tidy
add test checking we don t throw when creating genericindex with lshindexparams
sourcekit silence as wmisleading indent warning nfci undent the code that was mis indended leading to a warning with clang
clickhouse local use std filesystem equivalent
limit proc cpuinfo metrics to linux only
merge remote tracking branch origin master into master next
remove duplicate functions
fix getting read version while locked
clear messageupdate changes on item destruction
revert pausing in processqueuedpackets
update threadprofileevents cpp
update threadprofileevents cpp
open scene selected from quick open dialog
merge remote tracking branch origin master into master next
use is_equal_approx instead of vertex_snap when checking if ray is colliding with equiplanar csg faces
fix quit by closing non authed window
min max fix for standalone
remove change language link for secondary account
fix sharding key is not deterministic message
clangimporter adjust for llvm api change
js temp workaround for bloom binding error
sema force synthesized conformance to differentiable in emittedmembersrequest
always fallback to gtk icon theme to inherit icon theme even on wm based environments
update threadprofileevents cpp
macos export set correct external file attributes unix mode and creation time
save preloaded filters info
remove duplicate accounts toggle subscription
update threadprofileevents cpp
fix notification receiver name
fix proxies list box
refresh appconfig after login
yet another try
fix crash in update online on logout
write aces tonemapping in uppercase as it s an acronym
fix crash in redundant checkstarturl call
check type of filter for prewhere
fix possible crash in dialogs widget
check type of filter for prewhere
check type of filter for prewhere
fix crash in edited and replied message destruction
fix empty web page preview remove
fix a comment
reset cursor on window leave
fix crash in reading saved peers
fix tree s scroll_to_item not working correctly on some cases
resolve merge conflicts
add an else statement
resolved merge conflict
doc remove outdated comment in transactiontablepriv
add a warning message
fix opensimplexnoise get_image swap axes
merge remote tracking branch origin master into master next
don t handle cycles late if stretcher is active timestretcher seems to be broken
add documentation for new function
better rendering of example steps
fix arguments list in loadindex for histogram intersection
fix to upper case
wallet replace boost none with nullopt
fix transactiontagref construction
make timeline cursor in the bezier editor just as thick as the animation editor
address meng s comment
fix crash in password submit
fix possible crash in intro widget
rename getapplicationarea2 to getapplicationareasize
rename two functions in ns rename getshellevent to getshelleventhandle rename launchapplicationfromhost to launchapplication
install qt log handler
fix visual glitch on profile opening
allow to parse explain plan
gdnative support dynamic loading of ios frameworks
ignore blend times for inexistent animations
uncheck pressed if toggle_mode is unchecked
revert structure name change in state history abi
fix timeline cursor not updating on zoom change in the bezier editor
merge remote tracking branch origin master into master next
avoid breaking consensus while checking nonprivileged inline size
eliminate redundant condition check
fix possible crash in pip drag
kafka check that the data is still usable after parsing
fix laxist collision detection on one way shapes
remove lc converting to arrow
remove lc converting to arrow
fix test with ubsan
don t delay after incoming accept error
fix check that another artist with this mbid not already exist
break loop when the first focused window is found
add back conditional report
use global style options for drawing progress in transferlist
makes add new member dialog non exclusive for visual shaders
fix double gpu instance destruction
add test case for low surrogate handling
update statusfile cpp
speed up sync between primary and dr line allignement
update formatdatetime cpp
filemanger reset to root if directory is inaccessible
typechecker fix typos from previous patch
update arraysum cpp
fix graphedit reconnecting to disconnected port
fix string test code for mingw
memory set page table pointers before setting attribute memory
update aggregatedescription cpp
add comment to m412
fix editor crash when creating a custom resource from create dialog
clear a bullet area s overlappingobjects vector when remove an area from a space
imgcodecs fix test build with disabled jpeg and png libs
fix oppacity calculation if previous colour contains an alpha value credits paxxi
dx enable alpha blending for video renderer credits afedchin
fix matching video for gles
add comment about invariant master s committed version proxy s committedversion
fix gdscripteditortranslationparserplugin instance leak
dnn slice disable buggy ocv ocl implementation
add simple test for cbor byte
remove std replace
recalibrate reconnected controllers
temp residency followup
do not try to adjust memory tracker amount if it is not larger then in total
add skip list
min function corrected
skip list corrected
skip list documentation polish
skip list documentation polish
remove redundant if throw
fix crash on editor without a feature profile
apply subjective limit to context free actions as well
reset currentmetrics memorytracking periodically to the process rss
add a test for paddedpodarray just in case
add documentation for arrayfill
tweak plr debug
add cap runout co authored by julius ter pelkwijk mrseeker users noreply github com
get hostname without mutex
texture based thick lines only use textured lines for integer line widths
remove copyright from docs
address review comments
fix valgrind build
vulkan code cleanup
add assertions back and comments for splitmutation
remove and correct some comments
sema we don t have to explicitly set the type of the main function
fix typo in ibl computation comment
fix missing controller fan include
drop setinventorytxtosend based filtering
make sure unconfirmed parents are requestable
tested with trap function
fix exclusive child focus grab when there are more than two child windows
dbhub make url in progress dialog a bit shorter
g api change the default fd model in the privacy masking camera
nfc remove a redundant location print after dumping a decl reference
fix hashing in datapartwritercompact
project simplify launcher code even further
fix valgrind build
fix const return warning
remove useless code
fix refactoring bug
add additional empty check for the qstringlist returned by the installdialog
write through poco exception exception to mysql protocol client
fix leak in psidtostring on windows
fix compile errors
fix binaryop test ooops
fix floor ceil test with very small numbers
recreate externalauthenticators ldap on config update
dnn ie enable key_cpu_threads_num for windows
break out of scan loop if can t find adapter on first run
macos mono fix wdeprecated declarations build error
fix build for older versions of qt
update mergetreeindexfulltext cpp
update mysqlhandler cpp
update datatypenullable cpp
update prettyblockoutputformat cpp
ownership move ownership lowering past the eager specializer on the stdlib
address review comment
rpc print useful help and error message for generate
add comment regarding proxy tunnel usage in pocohttpclient cpp
gui fix weird font refresh issue
minor forced element styling removals
timewastedcounter styling remove
logstatuslabel styling remove
abstracttableview styling remove
dbg remove default exception skip limit
gui disable some debug output in release mode
wallet fix clang build in mac signed off by anthony fieroni bvbfan abv bg
recycle replace further alternative operators
set isscratchedspaceoverridemodificator to false when remove a rigidbodybullet from a space
csbindings delay key path application projected value binding until the whole constraint is considered fully bounded
fix the bug that forgot to break after setting the transaction option
csbindings improving comment explaning marking projected value as fully bounded
since we are using smart pointers an explicit destructor is not required
remove debug output
update registerdisks3 cpp
add errors if unknown params are used in a global or test section
follow device pixel ratio in trayiconfile
fix doc comment block
main function must return
avoid overflow when calculating visible_cells
add a comment to dd
fix overlappingobjects vector crash use clear_overlaps instead of clearing overlappingobjects directly
do not suppress wdeprecated for arrow
work around memory aliasing in descriptor parsepubkey
add todo comments
noopdriver now uses shadermodel gl_es_30 on wasm
frontend add missing diagnostics component to the default translations path
fix visual shader editor not clearing colour regions
properly pass safe margin on initialization fixes jitter
remove check for drop detached partition
fix stickers list display after account switch
correct is_reference attribute in api json for reference class
android fix display server always alerting no vulkan support
fix a boundary bug when throw cross module clear errors
remove bogus assertion with explanation
update hashjoin cpp
fix tensorflow onnx imports
cssimplify attempt unwrap optional base fix before root base mismatch on keypath application
merge branch bayesab of github com achimbab clickhouse into achimbab bayesab
dbhub do no show error message when download was cancelled
log avoid treating remote misbehvior as local system error
fix clang build
add rotate button for doc images to touchbar for media view overlay
add missing std string conversion
fix clang tidy
add comment to clarify the nullptr check
fix crash when closing a textfile
remove spammy trace
correct comment in this spsc test as a quick follow up to merge
mii manager resolve sign mismatch warnings previously the loop termination condition was testing variables of different signedness
remove duplicate config
fix possible crash in call top bar hiding
drop settings namespace
add cog in dict in wordladder cpp documentation
add handling of updates for edited media in scheduled messages
changed delay to change task priority to yield
dbhub add some line breaks to the input dialog text for cloning
fix isscrapedmbid value when another artist with this mbid already exists so neither flag or mbid being saved
add a wtxid index to maprelay
merge remote tracking branch origin master into master next
update profileevents cpp
ignore non wtxidrelay compliant invs
disconnect peers sending wtxidrelay message after verack
fuzz fix unused variable addrdb compiler warning
xp系统不支持菜单alpha blended图标 在为xp系统编译时不添加菜单图标
remove std move
add regression test
fix top scroll of compose controls on change of height
fix some lvgl warnings
prevent infinite loop in tree incremental search
fix directory make_dir and make_dir_recursive
replaced copy_with_new_args to clone_with_new_inputs
remove redundant new line
fix comments in beginning of file also improve error for unknown characters
better exception message in checkingsortedblockinputstream
fix crlf for clipboard
fix compile error
fix stuck of fieldheader when invalid link was provided
fix incorrect text in context menu item to cancel scheduled uploading
add esc shortcut to cancel edit in section of scheduled messages
remove display post views for sent forwarded scheduled messages
fix blurred display of scheduled photos that were downloaded too fast
add updating of webpage preview image on download finish
merge remote tracking branch origin master into master next
update new code in moduleinterfaceloader cpp for stringref std string conversion change
matching mat bed_temp condition
cssimplify handle generic argument for unsatisfied generic requirement bind constraint in repairfailures
use knob for remap cleanup window in write speed test
fix pipeline stuck for external sort
remove unused error code
fix crash on exit when compiled using mingw
fix error message about granularity
make mergetreesetindex checkinrange const
better exception during directories creation
always allow to save resume data in checking state
update moduloorzero cpp
update moduloorzero cpp
remove duplicate node update call
remove unnecessary override
serialization add more context to why we load swiftmodule files as volatile
merge remote tracking branch origin master into master rebranch
merge remote tracking branch origin master into master rebranch
merge remote tracking branch origin master into master next
fix build on windows
fix phrase translate of option for native window frame
update torrent status on move enqueued
fix build for older versions of qt
visualscript fix wduplicate branches warning when vsdebug is a no op default those branches did the same nothing
nfc fix grammar in repairviaoptionalunwrap comment
better error message
merge remote tracking branch origin master into master next
merge remote tracking branch origin master into master rebranch
add incremental description to usage messages
merge remote tracking branch origin master into master next
merge remote tracking branch origin master into master rebranch
replace stray tabs with spaces
hide overview widget now set appropriate splitter sizes
fix trace update after close file
replace a todo comment in examples
net log network activity status change unconditionally
actually test copy
fix test bug
fix ktx texture format for latest version
在网络连接中选择 选择全部 时 连接详情对话框中的序号不使用粗体
tesseract crash fix
fix modernize avoid bind in parallelparsingblockinputstream
fix handling of language switch for connection type in intro settings
fix handling of language switch for theme names in intro settings
make gdk backend limit more permissive
make ifunction executeimpl const
remove extra after member function definition
fix thread group for parallelparsingblockinputstream after remove std bind
updated text of fdbcli output
style fix remove unused declarations
boop the ci
keep transition value when replacing key
fix absolute value
fix file brief doc
add static to tofloat64
fix node hierarchy corruption when a node is set to parentless
enable copyforwarding on ossa
remove unneeded logging
fix crash on shutdown
fix clang compile error
videoplayer application crash exit when changing streams with rds stream present
cleanup use swift getparameterlist rather than hard coding it
revert abort on std out_of_range in debug builds
fix crash in web page preview display
remove redundant cast we don t need to cast away constness here it s already not const
gdscript allow new to be called in non static functions
sema remove on demand paramdecl checking for closures because they should have been type checked before the request
remove unnecessary code
fix bad endifs in filesystem cpp
wallet don t override signing errors
irgen method implementations for elided vtable entries are direct and don t need ptrauth
queue extra mix buffer
fix stream channel count when outputting to stereo
fix clickhousedictionarysource wrong access check
udp client remove unused boost include also silences a deprecation warning from boost on clang gcc
follow source operand
fix perf regression
better exception message
let the test pass on jetson
fix error style
set window focus when toggling with graph detached
fix ultra long node names
read tags from http sources but not streams
config make the save as identifier more consistent solves an issue with restoring the value upon reloading program
remove unused error code
remove default clause in switch for volumetype enum
braces style fix
style fixes and add unknown_raid_type exception to createvolumefromconfig
gcadapter only join worker thread if running joinable
update engine version
remove unused variable
remove settings apply when not local
fix arcadia build
remove thread limiting
fix reload_goal not being called when skeletonik3d start is invoked with p_one_time true
master rebranch fix up another implicit explcit stringref std string conversion
avoid recalculating string literal size on root tracker construction this is a tiiiiiiny performance optimization but it s free
master rebranch fix up missing include causing build failures
gdscript fix crash on await completion
adjustments add test result
pop from front to avoid infinite loop with nested classes
enforce correct type with printf style logs
csbindings nfc remove obsolete literalbindings vector
ast genericparamlist clone doesn t need to preserve source locations
fix header for totals and extremes in querypipeline unitepipelines
fix crash in columntuple updatepermutation
fix memsan error
fix next_pass materials not being drawn
optimize plan in explain
skip uinon step for usual queries
add limit to explain for mergingsortedstep and partialsortingstep
remove new lines
fix crash in incorrect passcode input
fix build failures cnk platform remove upstream a stringref std string conversion
fix an implicit stringref std string conversion issue in stringoptimization cpp
refine managementcommandsoptionsimpl clear range
change the number of colors when loading a bmp file
gui fix bug if you execute msgyn and then an error
dbg fix stringutils trim
minor perf improvement break ing out of a loop because it makes sense to do so
refactor use local argsmanager in cregtestparams
gdscript fix crash caused by inconsistent get_member
main silence fallthrough warning
update pipelineexecutor cpp
try fix tests
better logging on mutation skip
fix clang tidy build
merge branch master of github com clickhouse clickhouse
ignore constants for remote query from local replica
fix crash saving grayscale bmps
ignore constants for remote query from local replica
check if response is noperm with response ispermissionerror
show noperm error on key load
fix access to redis dictionary after connection was dropped once
add wiki reference
function builders fix typo in comment
add a comment of explicit flock in replxxlinereader addtohistory
add file comment for grvproxyserver actor cpp
change comment in diskcachewrapper cpp
fix textedit line width cache not being updated
merge remote tracking branch origin master into master rebranch
merge remote tracking branch origin master into master next
refix gdscripttranslationparser leak
calculate data stored by rocks
add missing close paranthesis for addtogroup
in g28 z is sort of known
ensure csg parent s _make_dirty is called when entering a tree
fix error codes
fix unit test failure
fastrestore short cut db for get and clearange
fix trivial bug in code
update tracecollector cpp
update pipelineexecutor cpp
update actionsvisitor cpp
cs address an outdated openunboundgenerictype fixme
update storagemergetree cpp
properly handle empty csgcombiners
fix window max_size acting as min_size
frontend replace some calls to getdeclaredtype with getdeclaredinterfacetype
fix heap use after free in the doctest main
set stylesheet of target widget rather then defining global stylesheet
fix the bug where we call old xff xff conf special keys
fix memory leak in test args
fix compile error
re add nowassigned
add missing paraetheses
fr applier received bytes per batch
transform csgshape collision when necessary
qt wrap tooltips in the intro window
function materialize should work for null as expected
fix arm64 build
fix function if with nullable constexpr as cond that is not literal null
rebootrequest uses seconds not ms
s fdbbench cpp fdbbench actor cpp
fix logging in mergetreedataselectexecutor for multiple threads attach to thread group
use not using primary index on part over not using index on part add primary
set instructionhighlightcolor to white
videoio fix compilation with aravis enabled
preserve initial state if push_lock throws exception
test repeat deadlock tests
fix including common config h in the arcadia build
attempt to fix arcadia build
fix error one more optimization
attempt to fix arcadia build
remove unused global variable from graph kruskal
gui call setwalletactionsenabled true only for the first wallet
packing honor thread count
reload bullet space override modifier even when rigidbody is inactive
place in anonymous namespace
download through tmp file in diskcachewrapper cpp
nfc remove empty check for singleexpression in refactoring cpp findfindconcatenatedexpressions
fix totals and extremes for delayed replica
update delayedsource cpp
add support to load sprites without extension identified only by their content
research how many characters users are typing before dismissing the cmdpal
add minor checks of ase header
replace ostringstream with reusablestringstream in consolereporter
report warning on fastrestoreapplierclearrangemutationsstart if delaytime is too large
addition to revert less indirection in fixedhashtable
update average cpp
typechecker nfc add debug logging before applying solution in applyfunctionbuilderbodytransform
remove get_output_as_single_output_node method
avroconfluent skip tombstone records
fixes leak with creating editor settings
gui capturing registers by reference instead of by copy in tracewidget cpp
fix up another caller missing second param of type dump
fix crash add new frames when there are tags present
fixes leak when creating bullet shape
fix build on msvc
update ownsplitchannel cpp
scheduler resolve sign conversion warning
add error message for empty grouping expression
localization add comments
reset transaction at the end of the test
fix timer leaks
allow scanning of album art served over http s
fix a week interval formatting for attach alter create quota statements
fix bug in onnx gather op
fix unit test
syntax fix type syntax tests for async
probably fix error
fix compile warning
fix function names
zydis fix isbranchgoingtoexecute for loopxx instructions
more optimization unit tests
demangle exception classname for std exception
pvr fix grid container model access with invalid index in cguiepggridcontainer getdescription
set medium priority for process signed block
update average cpp
fr samplebackup assert req batchindex controller batch size
net refactor clarify logic for selecting connections in threadopenconnections
verify interfaces by default in assert compilers isolated into its own commit so that it can be reverted easily
possibly fix clang tidy
using to replace typedef
update remote database information when an upload is completed
shorthand values for m575 b
followup for spi flash
fix build after merge
better error messages
if mutableconfiguration exists skip checking rawconfiguration in databaseconfiguration
db change column order on table creation
fix add spaces between numbers
docs minor correction
handle case newaccount in account query db
use a smallmapvector instead of a smalldensemap validatecodingkeysenum cannot iterate over a hashed collection without ordering guarantees
merge branch master into fixgraph
fix lint warnings
override cursor only if no buttons pressed
fix build on windows
macos fix crash on failed fork
revert change to readhotrangelog event
cssolver solveforcodecompletion should fail if constraint generation fails
merge remote tracking branch origin master into master rebranch
fix windows build
mark the first shape as inside not the second shape when csg shapes are co planer
try and increase fuzz coverage
win minor change to call coinitialize couninitialize
empty commit to restart ci
vulkan fix regression with window resizing
add a compact on deletion collector to tables
resolve review comments
update fdbserver workloads readhotdetection actor cpp co authored by jingyu zhou jingyuzhou gmail com
fix ctrl click not working
ie vpu refactor vpu configs
qt name rpcconsole executor qthread
qt name clientmodel timer qthread
qt name walletcontroller worker qthread
qt rename qt init thread before logging start
make drop table a little bit more robust
fix typo due to copy paste
correct fix for debug id logging
perfomed the requested changes
fix clang build
docs remove file name
fix gdscript analyzier not detecting resource subclass correctly
log error message
performed the requested changes
do runout reset after eeprom read
minor micro optimizations
pvr fix date format for listitem premiered and videoplayer premiered guiinfo labels
rpc assert that rpcarg names are equal to crpccommand ones zmq
bench allow skip benchmark co authored by martin ankerl martin ankerl gmail com
disable biquad filter
fix extui base methods
skip indentation of empty lines when indenting a selection
codecompletion fast completion is now supported in inactive if blocks
yuzu make use of qoverload where applicable eliminates a verbose function cast
fix incorrect foreground for read articles with custom theme
performed requested changes
fix performance no automatic move in interpretersmysqlddlquery
concurrency don t infer asynchandler unless concurrency is enabled
remove debug output
performed the requested changes
do not drop cache if drop rename in atomic database
irgen simplify typeconverter convertarchetypetype
ast simplify archetypetype requiresclass
merge remote tracking branch origin master into master rebranch
add jarvis algorithm
update tabseparatedrowinputformat cpp
dbg fix format function unregister copy paste error
fix renaming in memoryaccessstorage
prefer users from users xml in case of duplication
enable cxx interop is not a common frontend option
mergetree make exception message more clear
fix possible crash in call error handling
tests fix crash in regex tests
improve error message on unknown packet for client suggestion
improve error message on unknown packet in client
gdscript fix wrong argument check for formatting operator
gdsript prevent crash when completing unary operators
gdscript fix lsp getting wrong union value on unnamed enums
gdscript allow empty files to be valid scripts
merge remote tracking branch origin master into master next
g_message is a more appropriate log level for some messages
update pipelineexecutor cpp should have used back_edges instead of direct_edges
dont check process mode on open failure
fix handling embedded config
defer write throttling until found proper statistics for write saturation
repair vulkan after vkalloc upgrade
make xmlencoding tests slightly more efficient
lobster missed a test
fastrestore small fix on compile error
bug fix in kv byte counts
gdscript show error when function return type is missing
gdscript make subscript access be properly type checked
fastrestore send reply before assert fail
comment out command line arguments until atomics are supported
fix type error
track upstream llvm changes
fix lock order inversion when renaming distributed table
fix windows build
update tabseparatedrowinputformat cpp
update rowinputformatwithdiagnosticinfo cpp
improve comment also use push_back the lifetime of databaselockedkey and databaselockedkeyend will outlive trrequest arena since they re literals
only switch primary if extradb nullptr
fix crash in case of no tables in select
fix memory error caused by referencing a value with limited lifetime
fix merge issue that resulted in commit and gv latency band info not being published
throw special_keys_api_failure if clear called on xff xff configuration class
make clang tidy happy
fix typo exception name
fix network address order bug in exclusioninprogressactor
avroconfluent kafla skip malformed messages that do not contain at least the avroconfluent magic number and the schema id definition
fix main window drag by title on macos
increase fuzzing coverage of usockets
update avrorowinputformat cpp
refraction enable subpass usage flag this was causing a vulkan error when refraction was enabled
test pass empty tx pool to block assembler
test activate segwit in testchain100setup
add call to set up tracing
framegraph add graphviz labels for size and msaa
suggested changes has been made
remove extraneous defines
ast fix missing include in decl cpp
track upstream llvm changes
repeated constraints remove
skins process startup xml on skin change
fix build second try
add viewport subviewport node compatibility
try fix style failure
fix compile warning in fuerte compilation
unix_epoch_time fixup in rpc getnettotals
autodiff make differentiableactivityinfo dump print an end marker end markers are useful for filecheck
textord debugging invert default top bottom bounaries improve description
vulkan log driver version info this is useful to see which version of moltenvk is actually being used
update flow knobs cpp co authored by a j beamon ajbeamon users noreply github com
fix clang tidy
vulkan request the driver info extension
fix displaying caller address
in ctc_beam_search_decoder add a sanity check between input class_dim and alphabet
fix no space left extra info for poco exception
adds some info to the exception about zookeeper path
accept dynamicfontdata for fonts
util check for file being null in directorycommit
little bit more pretty
change rate calculation
fix compression codec
merge remote tracking branch origin master into master next
merge remote tracking branch origin master into master rebranch
update avrorowinputformat cpp
update avrorowinputformat cpp
fastrestore cancel actors when restore request finishes
merge remote tracking branch origin master into master next
update avrorowinputformat cpp
vulkan fix clear value list with msaa
fix vscode warning
merge remote tracking branch origin master into master rebranch
no com_pin needed for dwin
add ctrl i shortcut to actionstatistics mapped the ctrl i shortcut to open statistics in gui
net processing remove mempool argument from alreadyhaveblock
bench prevent thread oversubscription this change decreases the variance of benchmark results
update cachedictionary cpp
update cachedictionary cpp
remove read helpers
fix compiler errors
add wait to onerror
gdscript fix crash when completing code with signals
remove apparently unnecessary filter for cmo serializing references to globalinit once variables
slightly optimize very short queries with lowcardinality
gdscript fix crash when parsing properties
ast support silfunctiontype silblockstoragetype in clangtypeconverter this translation is a pre requisite to storing clang types in silfunctiontype
add comments explaining index_granularity_bytes safe guard
backupcontainer remove link to filename with random string
slideshow handle playpause action
csapply make sure that hole type are not propagated to ast
sema use getmetatypeinstancetype instead of casting to metatypetype
fix missing call diagnostic
sema add fixme for optional binding inference bug
sema remove unnecessary impact adjustment
fix sprintf overflow for the zero terminator
only use sudo when we are not user
fix codec in clone part
these log messages can quickly fill a debug log if accessing multiple settings
justify the choice of addr cache lifetime
remove uuid word for tables from logs
default codec if part is not stored on disk
fix compression codec in wal
add nullptr check to traceallinfo
update filedictionarysource cpp
disable support for pattern match of a whole machine address
nfc bridge based on silfunctionlanguage instead of representation
support for quorum
add logging whenever a wait failure client triggers
fix a whole screen flicker when enabling taa the history projection wasn t initialized for the first frame
allow directly read xff xff management without setting option
update aggregatingtransform cpp replaced redundant call to chunk getnumrows with local var
wallet refactor remove duplicate map lookups in getaddressbalances
ignore unknown messages before verack
make blank project name create an error not a warning
cssolver update expression after constraint generation while solving for code completion
rpc fix sign message verify rpc errors for invalid address signature
fix crash in schemaallowedhandler
ide fix typo to address unused warning nfc
addons an add on can not have a dependency that is a repo
add support for keys in cli
track upstream llvm changes
fix d bus type in mutterdbuslastuserinputtime
check if xcb screensaver extension present
update window shadow size on window state change
nfc correcting minor typo on a csdiagnostics comment
fix style check
fix compiler warning
fix empty columns case
cli create local addresses ports and scores report
fix compile error
change the remove missing button to disabled rather than hidden also set a minimum size
lastconnecttime was not being set for incomingconnections
remove spammy trace event
the workload sometimes would wrongly report a passed tests as failed due to not return after the hotkey is found
add validation of key types to ssd cache dictionary
fsync mergetree format file
fix rate calculation typo
apply review suggestion
stop query execution if exception happened in pipelineexecutor itself
gdscript allow extends to be used inside inner class
gdscript allow self to be used in class level
gdscript don t mark function parameters as constant they can be reassigned as if it were a local variable
gdscript properly set class inheritance for global classes
update fdbclient nativeapi actor cpp
fix prioritize marking a region as fully replicated over remove machine teams
support root on linux
support reshape with zero dim
fix filtering of search results in sectionedinspector
update obfuscator cpp
update storagesystemevents cpp
update astcolumnstransformers cpp
fix particles shader to use built ins
update local kv s contruction using result arena
doc explain address handling logic in process messages co authored by suhas daftuar sdaftuar gmail com
fix for lazy database
update storagedistributed cpp
silgen cleanup isforeigntonativethunk slightly bail early for the foreign case
fix pointer alignment
fix build useless binary
sema visit hoisted extensiondecls in extension binding
sema visit hoisted importdecls in import resolution
ast add hoisted declarations to unqualified name lookup cache
irgen emit hoisted declarations at the top level
always set repo origin to itself
use the correct row_limit_unlimited value
tools handle errors when opening the yaml file
tools unittests add missing include headers
unittests use assert_eq
fix mysql replication gtid event dump format
fix compare bug
refactor avoid duplicate map lookup in scripttoasmstr
null pointer dereference at gdscriptfunction call fix
make posvertex to vertex slicing explicit
main apply settings after applet configuration is complete
update schemas cpp
more formatting changes in querylog cpp
update querythreadlog cpp
try fix test
rawtransaction fix argument in combinerawtransaction help message
serialization refactor from large silinstruction loop writenooperandlayout handler nfc
merge remote tracking branch origin master into master next
fix style check declare error code illegal_type_of_argument
refactor remove unused header arpa inet h in protocol cpp
fix get_usable_parent_rect errors when initialising popup menus
avoid add margin twice along capsule y axis
fix alert introduced in sni example
videoplayer fix to use correct index for streams using a program number
fix tilemap tiles on horizontal vertical flips
avoid warning about harmless unfulfilled yields
add new feature show databases like
fastrestore fix trace event overflow in traceroleversionbatchprogress
adapt mlir dialect registration changes
fastrestore fix uninitialized variable in param asset
remove debug output
update executabledictionarysource cpp
clarify comparison precedence when enabling or disabling back buffer
fix expand collapse folder option not showing in filesystemdock
activate driver at console_user_server
add missing swift_fallthrough to the trycasttostring
remove superfluous whitespace before include
haiku build fix haiku is posix compatible
sil opt fix to satisfy all trapping instruction case at silinstruction maytrap
fix for clang tidy
check if old mouse column is still available
another fix for clang tidy
add comment explaining disks3 shutdown
change setting name
add custom null representation in tsv
merge remote tracking branch origin master into master next
turn off rocksdb in simulation
codecompletion remove unused forward declaration and fix up an out of date comment nfc
codecompletion remove unused field in codecompletioncallbacksimpl nfc
fuzzer reset default database before reconnect
remove false positive warning
move fallthrough one line up
gardening document a strange sort
make watch_overhead_bytes constexpr
update profileevents cpp
use the live data estimate instead of total data
don t set special_key_space_enable_writes randomly in fuzz api test
push pragma only for new gcc
fix missed database name when altering mv
remove enabling access_system_keys
update test of setclass
fix style error
fixes parallaxlayer offset when ignore camera zoom
try fix totals port for iaccumulatingtransform
merge remote tracking branch origin master into master next
minor changes to comment
create openal context without attributes
bump dns user agent
merge remote tracking branch origin master into master next
ignore unprocessed replication payload
fix unthrottle signal bug
fix missing clone in replace column transformer
silbasicblock a small refactoring nfc
fix rare lock conflict on checkprocessmode
supress strange warning
remove redundant change
only report if the field fullyrecoveredatversion exists
fix const warning
review some review comments from aj
add review suggestion
nfc consolidate sil langopts parsing
restore inputmode for module interface
always check repo for addon updates after updating repo addon
potential read access violation with an addon path setting
striped edges on node rename in scenetreedock
fixes crash when saving scene
creater user dir if non existant and pressing open project data folder
fix empty key segfault
fix create trimesh collision sibling transform
update code style
add more debug options
update interpreterinsertquery cpp
inline gatherutils algorithms
gui fix memory leak in cpudump and serachlistview
fix memory issue in test
fix rigidbodybullet areaswhereiam element shift
fix show hide fpu bug
fix double hotkey d
fix listitem addonlifecycledesc infolabel
break the query loop on unexpected_packet_from_client
add comments questions and todos
increase framegraph vector heap this is needed now that shadows are using the framegraph
set repo origin to self when enabling repo without origin
exception on double init of global thread pool
fix style check
better exception on non physical columns
init buffer in constructor
update threadpool cpp
fix parse error packet on event
fix simulation trace impl occurence
update threadpool cpp
add minv2 and andv2 into atomicops test workload
fix use of reference type as pointer in swift reflection dump cpp
do not check client s host when user comes from interserver secret
remove initialisms co authored by a j beamon ajbeamon users noreply github com
sema don t compute issimpledidset from getinterfacetype
sema bail out early when computing overrides of didset willset unsafeaddress and unsafemutableaddress
fix code style
sema fix closureeffects to pass the right declcontext for the lookup
update compressioncodecgorilla cpp
update compressioncodecmultiple cpp
update compressioncodecdoubledelta cpp
update compressioncodecgorilla cpp
update threadpool cpp
check stack size
get rid of query settings after initialization
update columnstring cpp
update functionsaes cpp
revert trace change now that blocking on a ready future is ok
loader nso remove unnecessary maybe_unused
when using copier column name as sort key with digital number as beginning will cause a syntax error
recognize more binaryop with constant
dynamic zookeeper config
update interpreterselectquery cpp
better drop replicated tables
better errors fix merge
update internals cpp
merge branch master of https github com hyazz clickhouse into hyazz master
remove previous modification
fix clang tidy
fix fall through in switch statement
change thread name for ddlworker executor threads
update joinsettings cpp
rpc add brackets to constructtransaction
apply symmetrical clear to txnstatestore for writerecoverykey
merge branch master into mlock code segment
netbsd implement os_unix get_executable_path same implementation as openbsd seems to work fine
update tablefunctionnull cpp
refactor istorage read with query plan
update mergetreeindexconditionbloomfilter cpp
resolve review comments
fix annoying arcadia
use initial_address for quota calculation when the interserver secret mode enabled
update mergetreedatamergermutator cpp
conformance checker eliminate redundant recording of missing witnesses
rename traces and add comments when getting empty worker list
fix bad code
fix clang tidy
fix clang tidy
update convertmysqldatatype cpp
update storagejoin cpp
update forkprocessor cpp got rid of an unnecessary local
replace null with nullptr in java bindings
replace null with nullptr in c bindings
update forkprocessor cpp
remove extra declaration
update renameat2 cpp
update databasememory cpp
merge remote tracking branch origin master into master next
update renameat2 cpp
use default codec instead of throwing exception
update threadpool cpp
fixup add missing comma
sema mark a patternbindingdecl as implicit in the builder transform
forbids varying declaration in particles and sky shaders
remove stray debug log message
more diagnostics in fuzzer
fix astwithelement clone
fuzz remove needless guard
fastrestore small fix compilation error
add a buggify so older api versions are run in simulation
add doxygen comment for receivemsgbytes
fix compile error
backed out buggify of apiversion because there are too many tests that depend on it for a patch release
add jitter to the delay
astscope assert that we don t re expand
window event listener do not use capture
fix pvs studio warning
check for null in inputmap _find_event
memory resolve a wdocumentation warning memory doesn t exist as a parameter any more
add optimize setting to explain plan
fixup use currline size for the first currlength in printoptionusage
zmq add support to listen on multiple interfaces
fr change applierremainmb map to unordered_map
fr fix typo for event fastrestoreapplerphaseapplytxndone
fix some pin inits
nfc add some todo comments for cleanup
remove unnecessary empty line from dictionaries
fix strange code
update processlist cpp
update context cpp
enable run loop profiler for test and multitest roles
fix msan report in querylog
remove unneeded includes
forgot to apply suggestion here as well
service restore unused function turns out this function is actually used but within a trace log
more forgetting duh
fix initialization order
migrate away from obsoleted functions
fileconfigure fails with assertion false failed
type checker eliminate a use after free due to c temporaries found by brent using asan thank you
csbindings start recording specifycontextualtypefornil fix when nil is bound to a hole
csbindings adjust impact of an event when nil is bound to a hole
keywords are case insensitive
loosen threshold for mali
add cfg tests
parser don t resolve decl references in the parser if the declaration has a custom attribute
draw selected item on top of everything else
filesystem shoutcast set image thumb from sam cast meta data
filesystem shoutcast set songtitle and artist from sam cast meta data
filesystem shoutcast set songtitle and artist from streamtitle
don t warp mouse when selecting tracks
fix light vector transform
fix install script
serialization gate clang type de serialization behind useclangfunctiontypes
astscope fix linker error when building without asserts
fetch longname when a file exists
concurrency fix new source file header
stick to bottom on new messages
reply in replies by double click
disable revoke all history for bots
inject discussed message in replies section
inject full album to the replies section
fix root message pinned bar shadow
don t read messages in channels i m not in
fix root comments post layout
fix a crash in failed sent scheduled messages
fix crash in mark folder as read
fix error message when exporting a write only property without a setter
fix tabbed selector display in replies scheduled messages sections
make the pressed autokey icon red to emphasize its recording status
update mergetreedatapartwriterwide cpp
simplify classdecl walksuperclasses it no longer needs to worry about circular inheritance
ast simplify classancestryflagsrequest
fix unit test failure
sema remove dead code from checkredeclarationrequest evaluate
fixes filesystem tree preview icon size on hidpi
fix event subscription in ddlworker
add count of active sessions as label of button in settings
connected popup menu display to archive button display in main menu
ownership move ownership elimination on the stdlib passed lower aggregate instrs
don t shadow local var
fix widget order in compose controls
fix crash in replies list management
merge remote tracking branch origin main into next
fix compile error
fix fdbcli would double count satellite dcs
merge remote tracking branch origin main into next
nfc documented irgenfunction indirectreturn
fix clang tidy
don t try to open comments in invite peek channel
update actionsvisitor cpp
fix root comments post layout
fix crash in main window destruction
fix crash in calls box
wasapi fix crash due incorrect ae format requested in some circumstances
fix crash on bad message in replies section
fix crash in sessionsbox when list of sessions is empty
net refactor move addlocal call one level up this change simplifies the following commit
keep system addon as system addon after update
make sure system addons always have origin_system as origin on boot
windowing gbm cdrmutils only drop master if we are master
filamentapp do not call postrender when skipping
remove beep while esc tab keystroke in function list
ast remove dead code from astscopedeclconsumerforunqualifiedlookup consume
pass context ptr to ref
update actionsvisitor cpp
allow block report from service message
allow delete all from service message
don t change window flags in window mainwindow recountgeometryconstraints
fix loading comments
fix build on linux
windowing wayland only build vaapi support if egl is present
fix thread num in handles view
rpc remove duplicate line in getblock help
remove unused logfilewriter ffs
update filesystemhelpers cpp
frontend disable parse time name lookup by default
return proper texture view format for decals
fix typo in visualshadernodesample3d generate_code
regex make first char before ep optional for single episodes
merge remote tracking branch origin main into next
another compile fix
increased the proxy commit memory limit
changed the recent requests to be the per second amount increased precision of cpu estimate
update main cpp
add an openbsd check
update main cpp
fallback to d bus methods if xcb based lastuserinputtime failed
matrix cpp setsize fix out of bounds access on cv mat steps
update fibonacci cpp done the changes as suggested
fix recording stop in voice messages
correctly use alcgetintegerv
aabb change updates bounding box spatial editor will now update the bounding box if aabb or transform changes
fix voice recording cancel
add option to remove snaptest retry limit
ast fix linker errors with the parser only build
check for global script class information before clearing it
use discontinuebackup in restorebackup actor cpp
fix compile error
add a more helpful error message when there is no current animation for the animation player
platform osx fix warning variable scope the scope of the variable real_path can be reduced
platform tvos fix warning
platform osx fix shadow variable warning fix cppcheck warning warning local variable share shadows outer variable shadow variable
fix selecting stickers in replies section
util do not use gargs global in argsmanager member functions
get back to moving args
merge remote tracking branch origin main into next
use errorcodes unknown_user
revert user not found detection in loginimpl and getidofloggeduserimpl
fix compiler error
fix numberparsingcheck to define found_invalid_number before simdjson h
explicitly throw error when snapshot has failed rather than hitting hard to trace assertion
fix unitialized memory
update fibonacci cpp
apply suggestions from code review
style check fix
add missing include
add logging for a scenario that should not happen
add a sample when a ping times out
add a suppressfor just in case a warning is spammy
revert change for compatibility
remove useless code
add missed mov extension to dialog files filter for album items
avoid unessary qstring allocation
declchecker explicitly compute captures for local functions in the decl checker
sema eagerly expand scopes in typecheckastnodeatlocrequest
avoid timing mishap warnings when mocking
make experimental capture work on windows with read write temp file behavior
fix channelgroups table create statement
cancel rename if file does not exist
fix an issue in uwp export caused by duplicate entry for extensions in content types file
update storagereplicatedmergetree cpp
addons dialogaddoninfo focus settings button
reapply avoid add margin twice along capsule y axis co authored by andrea catania info andreacatania com
better atomic operations on refcount
windowing gbm improve logging in cgbmutils
declare _canpipelinecommits to be const
prevent segmentation fault
style and nits for fee checking in broadcasttransaction
add wait statement
fix compilation error
adds pan gesture to statemachine editor
add retry loop for getreadversion in incrementalbackupworkload
sil don t print availability for unreachable unavailable functions
another response to review comments
update comment in visitcxxrecorddecl
avoid crashing if lldb imports c structs without enabling c interop
clangimporter import unavailable decls as unavailable
sema still use the current minimum deployment target as fallback
gui fix splashscreen crash when run with disablewallet
nfc lambda fy an assertion
update internals cpp
fixes httpclient poll crash after connection is self assigned
moved debug_advancemaxcommittedversion to right before the commit future is waited throw the se commit failure that was previously siliently swallowed
fix rescrape of single album or artist after changing info provider from context menu
peripheralcecadapter don t resume playback on tv standby if when the tv is switched off pause playback option is selected
more pixel affine coverage
addition to prev revision
fix premature end of animation playing backwards
update client cpp
update formatreadabletimedelta cpp
prevent error from delay actor
ensure grid index is valid before trying to change value
free a sortshader and a particlescopyshader
show android device name when connected only one device
fixes leaks with shader and material
expose all errors from all futures joint
remove g_simulator manual setting
addons no auto version pinning for origin_system addons
bind missing constant in visualscriptpropertyget
make txid delay penalty also apply to fetches of orphan s parents
unittest sema nfc switch to use attributedimport instead of deprecated importedmoduledesc
don t try to compare freed object to nullptr
try fix perf
remove unnecessary comment
merge branch master of https github com apple foundationdb into jfu fix snapshot backup agent
merge branch master of github com filipecaixeta clickhouse into filipecaixeta master
merge remote tracking branch origin main into next
net send post verack handshake messages at most once
eagerly move data out of sink
fix select count
wallet enforce sqlite serialized threading mode
implement sqlitedatabase makebatch
implement sqlitebatch close
implement sqlitedatabase close
fix possibly dangling reference when inserting into mv
no sleep when we have job
windowing gbm fix offscreen modesetting when gui plane isn t available
sema remove a bit of dead code
signet fix uninitialized read in validation
update storagematerializedview cpp
remove redundant logging
update storagematerializedview cpp
don t postprocess single parts
changes always show grid hotkey to prevent conflict with pan mode
addons retry repo update one hour after failed attempt not immediately
add error checking for a getreadversion call in a test
net update getnetworkname with all enum network cases
rpc update getnetworksinfo to not return unsupported networks
add action lock only if table exists
change the trace from sevinfo to sevdebug
increase the probability to generate keys after xff xff to test special key framework code
dump disjunction possibilities on multiple aligned lines for better readability
change error message in function if ifnull
fix unit test build
fix posix compile error
add more holders
check for cancellation after errorout senderror e
add comment for prioritystarved metrics metrics include prioritystarvedbelowx prioritymaxstarvedbelowx and prioritybusyx
move traceevent upward
sync fix texture_get_data bad barrier param changed srcstagemask to the valid stage for post transfer barrier
temporarily disable the load borrow checker a rewrite is ready and will be merged asap
update loadcopytoloadborrow for memaccessutils api
add errorcode to storageservertrackercancelled trace event
use traceevent error
ast add a no type parameters early return to findprotocolselfreferences
service bcat check client connection before interacting with socket fixes a crash when bcat service is offline
keep editor description metadata when changing a node s type also copy edit group lock when replacing a node3d
update math magic_number cpp co authored by david leal halfpacho gmail com
bpf lower the default memory settings
vulkan add missing field to vmavulkanfunctions
don t use the end of a window in calculating the elapsed time for roughness
sema don t desugar the type when diagnosing unexportable clang function types
nifm getappletinfo stub fixes crash for catherine full body
add remaining params
fix typo in connection cpp limiation limitation
explain why max_concurrent_queries_for_all_users exists
fix background moves start
bugfix chainparams add missing disabled taproot deployment for signet
rpc createwallet nicer error message if descriptor wallet requested and sqlite support not compiled in
gui create wallet nicely disable descriptor wallet checkbox if sqlite support not compiled in
blind performance fix
set capabilities with caution
addons fix set addon origin correctly
trying another fix
dvddemuxffmpeg compare audio channels in isprogramchange
remove unnecessary copies of errorcodetable
included referenced variable to avoid repitition
remove printing of windowpos screenpos in menubutton
adjust to llvm change inittargetoptionsfromcodegenflags requires a triple
fix texturesize texelfetch shader compilation errors
vaapi fix logging by remove trailing characters
reenable load borrow checker
nfc simplify saving of self to async context
nfc used consistent style for casting
update comment for disabling checkwritesenable in fuzzapicorrectness
add wait time after dd not found error
vulkan readpixels y flip workaround
update accesscontrolmanager cpp
fix build warning error
fix clang tidy warning
fix normal build
vulkan fix destroy while used for ubos
net add compat h header for htonl function
runtime tuple comparable associated conformance is not evenly aligned
fix typos in comments
align memory for fastallocator with valgrindprecise
remove redundant checking before remove
update accesscontrolmanager cpp
hide mark as read button in notifications when app is pass code locked
check actual socket path length rather than insnap inflatpak
fix replies button display in section of scheduled messages
close media viewer when photo message is deleted
update interpreterdropquery cpp
wallet fix bug when just created encrypted wallet cannot get address
fix parent memory tracker during query detaching
irgen temporarily treat async convention c as not async
nfc cached the layout in asynccallemission
async cc add todo for codesize improvement
assigning fpv_pawn when there are pawns in environment
remove unit test print
update permessagedeflate cpp
remove redundant false delimiter check
update permessagedeflate cpp
fix wrong exit code being returned
update threadstatusext cpp
fix performance test functions logical after move to clang
update description override in blob backup restore workloads
use anonymous namespace for backupcontainerlocaldirectory helper functions
start backup agent in restorebackup workload
addons dialogaddoninfo update button drop yes no confirmation
refactor remove use of boost algorithm replace_first
general use template deduction guides for lock_guard same behavior less code
gizmo handles transparent again
disable alt stack for sanitizers
enhancement for tileset sorting
update mergetreesettings cpp
fix build on mac os
merge branch better rethrow of github com yandex clickhouse into better rethrow
better grammar in comments ci rerun
doc missing comments for signet parameters
update columnconst include
don t stream videos when external player is used external player feature doesn t work otherwise
update expressionactions cpp
add verification_cooldown enabled check
test run assettest even if built with libs no
use long_ptr with setwindowlongptr
fixes property revert after saving instanced scene
update localserver cpp
update src interpreters context cpp
start backup agent earlier in restorebackup workload
prefer result from inverted image if the mean confidence is better signed off by stefan weil sw weilnetz de
update threadstatus cpp
merge branch sigaltstack of github com yandex clickhouse into sigaltstack
fix add to make protocol version incompatible
fix a typo
remove unused uid field inside paralleltcinfo
cdvdvideocodecdrmprime make hw decode setting visible if registered
fixes property revert for inherited child nodes
nfc used convenience to retrieve context arg
add table to system tables
fixes setting top_level not updating the global position
remove redundant metric increment
fix the regression of mark margin colour for dark theme
platform linux add libinput specific settings to startup
windowing gbm fix build on freebsd
try fix tests
doc update data directory path comments
async cc pass executor along to callee
async cc pass task along from caller to callee
fix karabiner virtualhiddevice manager app path
nfc add lambda to add executor filtered code for add resumeparentexecutor to the async layout through the new lambda
made find_set method efficient
update compressedreadbufferbase cpp
sil opt add an enable experimental concurrency option
fix compile warning about unused variable
gltfio fix segfault when consuming invalid file
vulkan minor fix for diagnostic output
diagnosticsqol modify diagnostic macro into single diag
add missing modification
add missing modification
decrease maximumtotaldata in writeduringread workload
update mergetreereadpool cpp fix comment
kernel process add missing ctime include fixes compilation on msvc
fix readability qualified auto in errorcodetoname
fix crash in tabcontainer in case of no content at all
update interpreterdropquery cpp
cplatformandroid register powermanagement
cplatformwin32 register powermanagement
cplatformwin10 register powermanagement
simplified the cast
make check style happy by not using db errorcodes in comments
fix errorcodes increment on error
eliminate extra strlen in errorcodetoname
fix readability container size empty in storagesystemerrors cpp
do leak check before stopping network
synchronization clean subpass dependency init fix incorrect subpass dependency initialization
add information to silverifier stack nesting diagnostic
change macro to diag in storeddiagnosticinfos array
allow fetching duplicate parts to detach
silgen fix an unused variable warning
post merge cleanup
doc document that wallet salvage is experimental
update parsedatetimebesteffort cpp
restore ui hidelayer call in showinfolder on linux
allow dropping any amount of any files
fix crash when add more files to sendfilesbox
show view x replies in all groups
fix build on macos
fix pinned bar consecutive animations
fix pinned message of self destruct media
improve pinned messages title
fix sending files in scheduled messages
try creating base dir in filewritedescriptor
fix cancel of album parts uploading
fix go to original button in outgoing albums
update replicatedmergetreequeue cpp
clang tidy performance inefficient algorithm
log the difference between process rss and memorytracking metric
allow ctrl middlebutton to copy rva
fix a crash in _dbg_addrinfoget thanks to justasmasiulis for reporting
fix build error
disable active editors when node gets deselected
merge remote tracking branch origin main into next
revert change for storeddiagnosticinfos
nfc constraintsystem update the documentation for existingoperatorbindingsfordisjunction
fix crash due to unreachable code in properties
add a new example to check correctness of the code
update mysqlgtid cpp
try to workaround msvc compiler bug
do not load symbols when getting syscall indices
do not set ntterminateprocess breakpoint per default
vulkan disable unreliable blit diagnostics
changed moment of stream object initialization
remove unused function
core initialize gpu before services
service hle nvflinger fix potential shutdown crash when gpu is destroyed
refactor qt use vqueuenotifications clear
use ibus portal whenever it present
merge remote tracking branch origin main into next
fix caption comment label in sendfilesbox
fix typos on injected params for producer_plugin log_failed_transaction
remove redundant include
vulkan adjust usage flags to allow depth blits
fuzz fix decodehextx fuzzing harness issue
fix leak in test code
addons install no toast notification for dependendcies
layout loop unrolled blocks next to the original loop
looprotate avoid introducing new critical edges during rotation
remove redundant check
fix build error
update expressionanalyzer cpp
try fix header for ttlblockinputstream
fix response serialization
check max bytes on disk
update actionsvisitor cpp
fix a crash found by nightly correctness
bluevk improved error messaging
change naming of keypoints comparator
the trace interval was never begin and thus commenting out the end call to fix the assert failure
consider async during classification of call sites to rethrows functions
windowing wayland fix spelling error in egl guard
replaced null with nullptr
fix bug when merging classifications that was clobbering isasync
add isasync param to forrethrowingonly for consistency with forthrow
fix copy paste in pvrclientmenuhooks cpp
update context cpp
fix inconsistent whitespace
remove obsolete code from join
make it easier to reason about node eviction by remove unused nodeevictioncandidate addr caddress
update reinterpretas cpp
update aggregatefunctionorfill cpp
imdrawlistclipper avoid over reserving memory
wallet upgrade the chdchain version number when upgrading to split hd
fix wav resources ignoring the audioserver s global_rate_scale value
fix jni binding
take out unnecessary includes
fix lib clangimporter importtype cpp
fuzz assert expected decodehextx behaviour when using legacy decoding
update fdbserver logrouter actor cpp
add helpful placeholder in command dialog
set bilingual error completely
qdir temppath already has fallback to tmp
fix ibus portal service name
fix crash in search row select in privacy edition
cancel search in chat doesn t reset search query
fix typo thanks to justanotheranonymoususer
fix preview display of small media in sendfilesbox
remove unused phi define in math funcs
update complie expressions
update complie expressions
update complie expressions
update complie expressions
wallet set databasestatus success in makesqlitedatabase
update complie expressions
update complie expressions
fix choosing directories in snap and flatpak
fix areqtpluginsbundled to include static binary
update complie expressions
try fix integration test
fix uninitialised variables in the basematerial3d
try fix integration test
remove unnecessary header
update threadpool cpp
update astcolumndeclaration cpp
merge remote tracking branch origin main into next
update mergetreeindexconditionbloomfilter cpp
update mergetreeindexconditionbloomfilter cpp
add one more case
fix bad wording
remove unneeded dir
fix improper error handling
fix ip subnet comparison
artwork fix season all posters and manage context menu item
remove unused code
set parent window id for portal autostart dialog
fix getting filechooser portal version
use kiconname on icon creating
allow cancel search in chat and keep search query
fix quit from fullscreen on macos
fix duplicate variant constructor typo
fix special build
make member function const to prevent readability make member function const
try fix tests
fix gdnative build when webrtc module is disabled
remove attempted to free invalid id error
fix minor clang error
update expressionactions constructur
update expressionactions constructur
higher mmap threshold in debug build
fix an exception with export functionlist while absence of config xml
fix strange code in interpretershowaccessquery
avoid signed integer overflow when loading a mempool dat file with a malformed time field
try fix tests
try fix tests
minor change in query profiler
merge remote tracking branch origin main into next
disable tracecollector if trace_log is disabled
set thread name for the tracecollector
nfc downgrade compilerinvocation to frontendoptions
stop offering gleebug for beta testing users can switch from the settings if they want
update lastexceptioninfo on breakpoints
fix transparancy in the stdicontable
fix compile expressions
fix cast not showing when play next is set to episodes
since changing the ibus portal check it is not compatible with snap anymore
fuzz use consumerandomlengthbitvector in src test fuzz connman and src test fuzz net
release pressed action if event is remove
update install cpp
update connection cpp
make description method const co authored by lukas joswiak lukas joswiak snowflake com
bugfix co authored by lukas joswiak lukas joswiak snowflake com
remove dead code
fix clang tidy
update connection cpp
nfc add a typebase conversion constructor to nevernulltype
dialogs cguidialogkaitoast addtoqueue avoid duplicates in a row
nfc drop unused includes from frontendtool
fix bug in git import tool
async cc add asyncfunctionpointer to wtable
async cc add asyncfunctionpointer to vtable
use unlikely instead of boost_unlikely
update connection cpp
update programs install install cpp
gdscript fix native class not set with inheritance
fix posix compile error
fix clang tidy
fix android export jarsigner error with import whitelist
add comment restart ci
don t generate machine id if it is set by the user
nfc remove unneeded void cast of clangdc this is just a simple cleanup
move the histogramreporter to actors list
attempt to fix windows build
silence an assert
handle unowned and unmanaged references
clarify a comment in memaccessutils
silence false positive gcc warning
explain waitforversion why wait for version minus max_read_transaction_life_versions
add missing optional h include
fixes crash if vulkan presentation surface is not available
update avrorowinputformat cpp
remove redundant anon namespace in clickhousedictionarysource cpp
fix issue causing export all to fail
fix new transform to pixel snapping logic co authored by lawnjelly lawnjelly gmail com
gui send cursor data from textedit
fix custom loader saver broken
subtitles ass add events as possible parser option
fix rocksdb_error value
controllers npad load input devices on init
fix typo in the installlauncher cheat code
use only really supported icon sizes
re check the screen media viewer appears on before adjusting geometry
add nd gmatdesc serialization test
update interpretercreatequery cpp
remove color space before sending in jpg
irgen squash an unused variable warning
add code coverage macro to datadistributiontracker safeaccessor operator
swift ide test disable parser lookup
fix origingeopoint gps error
remove stale comment
add code coverage macros to pagechecksumcodec checksum
sema don t diagnose forward references to debugger variables in pre checking
merge remote tracking branch origin main into next
add empty commit
add show animation to voicerecordbutton
update src interpreters ddlworker cpp co authored by tavplubix avtokmakov yandex team ru
order inputs for actionsdag
skip unused keys
update columnconst cpp
add converting logic to actionsdag
fix typo applicatioin
add converting logic to actionsdag
try fix asan build
add converting logic to actionsdag
remove converting transform and step
don t write traces to log file in simulation
explicitly inline trivial function in cumulativereporterbase impl
add newline fix initialization in constructor
remove converting transform and step
fix deletion of nodes with exported node paths
update replicatedmergetreemergestrategypicker cpp
fixes rotation in select mode on macos
don t set screen for media viewer on wayland
merge remote tracking branch origin main into next
fix sending bot commands from autocomplete
wallet refactor getclosestwalletfeature
fix mock sqlitedatabases
gui force descriptor wallets when bdb is not compiled
rpc require descriptors true for createwallet when bdb is not compiled
enforce salvage is only for bdb wallets
add comments about error type change
allow to open multiple projects when some are not imported or without main scene
join processing thread
check uniform set validity before freeing
fix bug in rolled trace code
fix windows build datetime to use utc
update visiblewidth cpp
fix zipalign command name on windows machines
fix test_model detectionoutput
fix stupid error
fix legacy group service message in chats list
scroll to bottom when sending a poll
use next in send media box in scheduled section
fix reply stuck display at sending voice in replies section
remove delay for voice lock widget appearing
fix unwanted flickering of record button when recorded data is empty
cdrmprimetexture only create the texture if it s not valid
less verbose logging when fetch is impossible
fix build break with previous merge accidentally reinitialized a variable in the previous merge with https github com microsoft airsim
fix heap use after free
fix memory orderings
actually use privatized arena
fix merge distributed with join
remove the incorrect unnecessary versioning stuff since the error is thrown by the server
sil fix ownership of getcurrentasynctask builtin
add dirty hack for connections close
fix clang tidy
olsc move member initialization to after member functions
swift reflection dump fix resolve call on branch
nfc remove unused modulecachedir variable
revert user directory name change
fix ip dict build
remove unnecessary condition will never reach
initialize client and flow knobs
fix camera2d rotation with non square zoom
update zstdinflatingreadbuffer cpp
update zstddeflatingwritebuffer cpp
make warning about reference singletons more accurate
add accept_event calls for scenetreedock and basebutton
fix clang tidy message
svc remove unnecessary maybe_unused tag the parameter is used in this function so this suppression isn t necessary
update storagememory cpp
conditional msg for missing lm binary add
fix cast not showing when play next is set to episodes
update zookeeperimpl cpp
update zookeeperimpl cpp
update zookeeperimpl cpp
add eof check in receivehello
fix potential int64 overflow in int array conversion
remove unused profileevents
move trackercancelled higher in catch block
cpowermanager check settings for nullptr
videoio add missing getcapturedomain methods
update clear button when clicked
ceglimage allow logging if logvideo is enabled
member initialisation coverity fix
update executequery cpp
show loaded tls library info
update src datatypes datatypetuple cpp
update registerdictionaries cpp
qt improve comments in walletcontroller getorcreatewallet
remove double space
gdscript fix return of cast expression on compilation it was mistakenly returning the source instead of the result
fix formatting of settings clause of the insert query
hid check if applet_resource exists in initializevibrationdevice
fix prints and printt functions printing as errors
remove complex code on session timeout
update install cpp
increase sleep time on a test case to avoid spurious failures
properly check distinct columns
properly check distinct columns
fix clang tidy
sync const qualify the argument of double_lock_detected it is not modified in the function so should be const
sync use hasreason in double lock tests hasreason is shorter than a lambda function
add a null check in logpostfixdummy
update storagememory cpp
remove redundant call of handlenewstream in streaming executor
fix issue for assert
do not remove arrayjoin from actions
fix unitialized memory
handle rename failure in dumpmempool by using renameover return value
fix data race on global blockstreamprofileinfo in pullingasyncpipelineexecutor
uwp fix typo in kernelversionfull string
minor changes about old layer set layer group
tests fix warning about comparison between signed and unsigned
mention lack of optional setter arguments in classdb add property
make pullingasyncpipelineexecutor thread safe right now if currentthread attachto throws it will not catch it
fix status of torrents without metadata
don t try to remove folders for a torrent without metadata
common fiber use virtualbuffer for stack memory this will be aligned by default and helps memory usage
resolved conflict add new compression method in http
show editable children in the connect dialog
reduced margin on popupmenu
port clickhouse code to boringssl
fix missing function hints
add missing return
pvr fix hide no info available epg setting
make sure autodifflinearmapcontext is called
update page value properly of scrollcontainer
properly handle append extension option changing
fix accidental stop of voice recording in sections
add colors overriding for call settings button from mute button
update renameat2 cpp
fix writeuints bug
update a comment in escapeanalysis to reflect the most recent compile time fix
fix style in ipv6tobinary cpp
add comment for ipv6_masks_count
enable incremental usage for fdbbackup on command line
fix pck lookup for extensionless binary in macos resources
add check for systemlog tables engine definition
add log for systemlog tables engine definition check
xatlas should be using the options configured here it was using the defaults by mistake
update fdbserver tlogserver actor cpp co authored by a j beamon aj beamon snowflake com
remove word ghetto from repo
ignore hint entries in game controller mapping
update storagereplicatedmergetree cpp
merge remote tracking branch origin main into next
remove unused imports
fix ugly bug
fix missing import
do not reject blind writes i e empty read conflict range
add code coverage macro to processpoprequests
nfc remove unused variable loc fixes a warning for unused variable in ownershipmodeleliminator
add check for existence of databaserebootworker in c api
merge remote tracking branch origin main into next
sil don t verify functions whose body was skipped
update queryplan cpp
add driving joystick type to windows joystick handling
hotfix check in_memory_parts_enable_wal
tables return false when window is collapsed consistent helpful for doc fix empty context menu
tables fix headers closing popups
tables non resizable columns also submit their requested width for auto fit
tables remove dubious window skipitem assignment in tableautoheaders
tables fix assert crash when a visible column is clipped in a multi clip group situation
tables fix lower clipping when using imguitableflags_nohostextendy
tables fix stacked popups incorrectly accessing g currenttable of parent in stack windows
tables fix tableheader not declaring its height properly do not declare width
tables fix auto fitting columns not marking settings as dirty fix cellbg fill not narrow clipping on scrollx freeze
update aarch64subarch_arm64e triple name to match llvm
fix swift reflection dump cpp for recent llvm changes
merge remote tracking branch origin main into next
merge remote tracking branch origin main into next
update fdbserver masterproxyserver actor cpp co authored by a j beamon aj beamon snowflake com
fix unhandled exception re thrown in the editor
revert change to forcerecovery
ptrauth auth parent async context after loading
nfc use functionpointer getrawpointer for vwitness
frees canvasshadersdf reference prevents warning at startup
fix background color in calls topbar for force muted state
hle kernel svc remove unnecessary hack in svcsleep
subtitles ass fix wrong screen position for fractional pts muxed ass subs
exif fix gps altitude values float
don t leave call when joining other call from other device
complex text layouts fix missing itemlist item text buffer init in add_icon_item
avoid type conversion warnings in plot dock
boxcat avoid unnecessary object copy
add diagnostic information when two merge tables try to read each other s data
avoid add playlist etc items when in playlists folder on music playlist editor screen sources panel
wallet document that tx in createtransaction is purely an out param
fix double counting of range reads in transactionmetrics
update interpretersystemquery cpp
nfc drop the unused interface key from the use enumerator
nfc delete an unused entrypoint
controller use std move within converttofrontendparameters avoids unnecessary copies
gui controlgrouplist update scroller sizes when list size changes
use unique_ptr for g_tracer
memory resolve wdocumentation warning for write write doesn t return anything so the returns tag shouldn t be present
detect more dark themes on linux
android fix mouse capture relative wrong
merge remote tracking branch origin main into next
add additional size check in debug mode
fix trying to set grid visibility on an invalid instance
lsp fix iterator in enum api dump
fix ubsan issue
asset library scroll up the scrollcontainer after page load
add throwing exceptions
init use makeunique to construct peerman
merge remote tracking branch origin main into next
updated if using old settings xml default them to level basic to make them always show
fix scene being modified when arrows are pressed with no node selected
sil teach the verifier that an upcast of an address isn t a consuming use just a missing case
merge remote tracking branch origin main into next
fix build without gtk
videoio turn off syncronized sink in gstreamer
clarify comments around outbound peer eviction
fix tooltip position in embedded window
merge remote tracking branch origin main into next
fix object check on variant key checker
don t serialize dynamiclookup dependencies in modules these are effectively uses from the perspective of a client module
add rate control on both db file and log file
net processing remove unnecesary nnewheight variable in updatedblocktip
wrong double quote output with csv fix
decrease verbosity of disconnecting clients right now they go to error log
fix master branch merge artifact
remove diagprinter directly from the compilerinvocation
display the number of results for global search
clibinputsettings fix nullptr check
ceglimage check if eglattrkey is valid before comparing
add host specific include for getpid in frontendtool
suppress the output of the printingdiagnosticconsumer when in frontend parseable output mode
update encodexml cpp
fix misleading comment about call to non existing function
update storagesystemstacktrace cpp
have to resort to no_sanitize_thread
another try of tsan fix
add maybe_unused for msg parameter in handle_error_code
merge remote tracking branch origin main into next
fix editorpropertytext change signal emission
fix struct initialization
python fix to allow pydev debugger to work
cosmetic replace xbmc leftover
fix style issues
rpc allow getblocktemplate for test chains when unconnected or in ibd
add parse error if no select after as keyword
call movetoscreen from handlevisiblechanged in overlaywidget
update arrayaggregation cpp
one more time
fix visual glitch in replies section with tall input field
fix an issue with sizeofimage not being page aligned in memory map
support for lazy database
allow skipping more bad packets after a good one
check if cjdns address is valid
fix clang build issue
update src formats jsoneachrowutils cpp co authored by tavplubix tavplubix gmail com
remove csv restriction
fix rpc return type
fix rpc return type
make find in finder result s text direction follows the search results one
fix freeze on screen disconnect
gui fix issue editing expression using gui changes the watch type back to uint
fixes after merge
rename the confusing menu option delete to close in the custom tab bar
update interpretershowcreatequery cpp
try fix data race
fix a typo in a comment
gui define max_digits_btc for magic number in bitcoinunits format
fix unimpl unity methods
remove g_isopencvactivated assign and clarify
sema add a timer for precheckexpression
fix infinite loading when no members in the group
try fix integration tests
mmapreadbufferfromfiledescriptor use getpagesize
fix wrong error code
fix gdscript and operator
fix bitwise and eval not updating return type
fuzz call sendmessages after processmessage to increase coverage
better exception messages
fix style issues
reused result type
merge with master
disable trap back
fix style check fail
suppress gdb warning
fix crash parsing a serialized reference
simplify the code
fully disable session manager interaction since session manager functionality is not used anyway
fix bad sleep
clean up logging of outbound connection type
update multiply cpp
fix a code comment typo
update install cpp
update install cpp
need proper locking
put thread back to set if it had stolen all tasks of backoff thread
fix mouse keyboard input on windows
fix style check
fix gpu extension conditions
update irowinputformat cpp
fix broken members panel in visual script editor
fix style error
fix group call bar in legacy groups
turn off robust buffer access by default it can be a performance bottleneck in some hardware
remove default commit message
update datatypearray cpp
sil combine eliminate a dead parameter we pass in deadendblocks into sccanonicalize s constructor now
exit clickhouse when failing to initialize zookeeper with logical error in ddlworker
update columnscommon cpp
copy the network address stringref
gframe as internal node in gexecutor
remove logs from thread fuzzer
remove more changes
fix special build
refactor code a little bit more
rpc add missing description of vout in getrawtransaction help text
update threadfuzzer cpp
merge branch arrays as nested csv of github com yandex clickhouse into arrays as nested csv
codecompletion zero check interval should always check dependencies
fix log message for memory tracking drift
fix crash when iterating through empty dictionary
more fuzzer fixes
ast simplify extendednominalrequest writedependencysink
fix variable names in asynchronousmetrics update
revert unneeded changes
use deprecated pango_fc_font_get_glyph for old pango versions signed off by stefan weil sw weilnetz de
fix possible crash in app shutdown
use langpack strings in a better way
show better tooltip for force muted in voice chat
fix highlighting of self row in voice chat
fix darwin build
fix optimize deduplicate
fix for uninitialized thread pool
fix filename disambiguation on scripts in certain occasions
fix text for fonts_dir parameter signed off by stefan weil sw weilnetz de
add terminating linefeed to fonts conf signed off by stefan weil sw weilnetz de
fix clang tidy
update treerewriter cpp
update parallelparsingblockinputstream cpp
support multiplevalueinstruction in swift getinsertafterpoint
update istorage cpp
fix the string get_base_dir logic to properly check for top level directories on windows
fix invalid invocation of get_class_loader the call was made on a godot instance instead of an activity instance
fix mistake in gltfmesh mesh property
fix for review
fix crash in legacy groups speaking typings handling
fix add clamp for iou_score
do not use watchdog when server is run from tty
removes the gdnative library when no script gdns references it any longer this enables hot reload for gdnative
fix bug no newline after exception message in some tools
fix multi pin bar render after theme switch
don t offer sending pdf s as photos
update parsertablepropertiesquery cpp
update parsertablepropertiesquery cpp
fix possible infinite recursion in video calls
boop the ci
test set correct nvalue for multi op return policy check
fuzz check that null_data is unspendable
implemented simexternalconnection getpeeraddress
implemented simexternalconnection read
remove invalid assertion from simexternalconnection read
reduce sleep in simexternalconnection write
ensure flags are applied to cvtt options
more correct error code
fix unexpected change
add min word size
save fastresume when setting torrent speed limits
fix etc quality setting
update decodexmlcomponent cpp
fix build with disable graphics signed off by stefan weil sw weil de
fix incorrect title and button text in inherit dialog
mark tcmachineteaminfo size const
fix schema test compile error
add body to private copy constructor copy assignment
add return statement comment
also initialize class member
try fix test
try fix pipeline stuck
try fix pipeline stuck
try fix pipeline stuck
add toggle for character panel history clipboard in shortcut mapper
fix layout menu not working
udp client process packets only for the correct pad
add missing normalized accessor property to gltf document
fix layout in intro settings
sw do not build unit tests by default
fix nullptr assignment in tessbaseapi signed off by stefan weil sw weilnetz de
update src api tesseractmain cpp co authored by stefan weil sw weilnetz de
hle service ensure system is powered on before writing ipc result
core kernel clear process list earlier
fuzz check that certain script txouttype are nonstandard
core do not reset device_memory on shutdown this will be reset on initialization
core hle kernel clear process list on boot
fix exit code of watchdog
check marks fix comments and help message
check marks add decompressed size
compressor remove superfluous warning suppressions
fix trie s word sort comparator
service pcie fix invalid initialization argument
fix hiding cancel button in state of listen to recorded voice data
disabled message editing while voice recording
make openal debugging easier
correctly show legacy groups with no admins
remove unused function comparestring signed off by stefan weil sw weil de
remove unused local function charcoveragemaptobitmap signed off by stefan weil sw weil de
don t allow kicking yourself from legacy group
update compressor cpp
update keycondition cpp
do not clear materials when setting mesh
gdscript crash at incomplete const bug fix
more correct words about parser
remove useless support for symbolic port names
send fatal logs in all tests
sw add new unit tests
merge branch remove getservbyname into harmful
fix linux build
fuzz remove duplicate all_output_type array
powermanagement onsleep send pvr to sleep after stopping playback
audiotrack disable multi channel float once again
do not iterate over map when remove its values
update translatequalifiednamesvisitor cpp
main tidy up enum comparison enum classes are comparable with one another so these casts aren t necessary
sw remove deleted file
compressor remove extra check for seeking of input
