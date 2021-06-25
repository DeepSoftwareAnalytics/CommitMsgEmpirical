isnan wasnt compiling
fix current scene run when scene saved
remove unused nextargidx operation
create configuration directory at launch
merge remote tracking branch origin master into master next
add file storage release
disable ast logging
changed fetching the video sources once is enough while cleaning up the video library
silgen add an assert that formalaccesscopy is only called in a formalevaluationscope nfc
fix pluginscript add_global_constant mandatory value check
fix linker errors with wifimulti
allow m110 to handle a second n argument
explain test subtlety
layercolor initwithcolor should return false when layer init returns false
add missig cygwin test
service remove unused requestparser instances these aren t used so they should be remove to reduce compilation warnings
dvdvideocodecdrmprime release video buffer before receiving next frame from ffmpeg
fix android template install validation
fix the missing dynamic axis for customproxyop
fix compiler warnings
arraysemantics simplify the code a bit nfci
remove useless debug message s
fix playing games with videoplayer when emulators are disabled
ensure there is a datafile info entry for each datafile of a collection
linux warning fix
error if compiling spdlog cpp under header only configuration
scraperurl get fix avoid race conditions
use getdynpath for getplayingfile
regex really works
add more comments
includecancelled for workersnapcreate
pvr fix refresh container after view mode is changed in guide window
peer review changes
improve parameter description
clear previous cancelled flag
qt add try_lock back to peertablemodel
typechecker propertywrappermutabilityrequest should not return none when parsing getter setter from swiftinterface file
diagnostics detect situation when argument would match if it wasn t optional
changed background color of saved messages button in main menu
use std thread hardware_concurrency instead of our own solution
merge branch master of github com lemire simdjson
gardening document a strange sort
make random more random
reorder variables to comply with the auzure build pipeline
changed log topic label
fix system exit status
don t set non existent sled_pin
settings don t ignore pvr channels in savefilestate
clarifying getrawtransaction time get help text
show sdl error message on init failure
fix right click menu position in seeding list
when clearing art during refresh make sure we clear it all
wrap function into anonymous namespace
rad deg correctness
fix interlaced modes on fullscreen
videoplayer fix a v sync correction for clock feedback mode
fix user extensions path on windows
revert temporarily disable an optimization to avoid some crashes
set last error state when a cmd error happens
update complie expressions
adsp remove wrong tmsg_setaudiodspstate call
silgen fix a warning
split always support packing
make most_recent_compact_block a pointer to a const
change utest when demuxer use io interface
fix comparison bug with inputevent
gtest fix compile missing include file
use descriptive constants more
minimize to tray only if the relevant option is enabled
reduce grace period
fixes broken scene created via new resource
avoid inertia calculation for empty shape bullet
fix parse error packet on event
changed copy paste error message
slighly ugly handling of nester curop
update isconstant cpp
pvr make cpvrmanager updateitem more failsafe
fixes import of models as packedscene
cguiwindowslideshow use cguiviewstatewindowpictures getextensions as a fallback if no file extensions were specified in runslideshow
disable auto tmm when not using default savepath from monitored folder
fix a description of two methods
do not retry connection as often for benign_other disconnect no blocks
add include float h
fix windows compile of getenv facilities
remove duplicate comment
compaction bug in agency
remove unused variable which caused linux build to fail
addon settings parse action data
do not throw in destructor
merge branch master of https github com telegramdesktop tdesktop
fix unused variable warning
make changeowncustomdata privilege target the user s database rather than the cluster
fix mongos test
dbg remove restrictive heuristic for string detection
sema fix a couple of nits
add list of unsupported commands in console
increase timeout on killing
fix style check
remove useless code
check return value of ao_play
dnn slice disable buggy ocv ocl implementation
ctrl f hotkey for the find window
add positive list of nodetypes
elim warning on windows
fix trackers remove
hide download url column in search results
flow platform cpp
remove short option name for active snapshot interval
give the agency more time to reply before we retry
fix exit confirmation dialog centering
fix removal of dllloader exe
modify wrong codes
uassert instead of verify for invalid skip
investigation of performance degradation of jemalloc
make sharding work with orderby
fix out of bounds error when we have a constant
service add missing headers inclusions where applicable gets rid of a few indirect inclusions
remove reference to bug id
wallet disable free transactions when relay is disabled
serialization manually write finalattr if isfinal
avoid using n in logs because it will link bogus issues and pull requests when posted on github
correct scrobbler submission logic to honor settings
trivial fix typos tempoarily temporarily inadvertantly inadvertently
caddondatabase invert enable check and include checksum to ensure invalid versions are not included
service restore unused function turns out this function is actually used but within a trace log
remove dead lowering path for inout elements of tuple parameters
rbp add missing timingconstants h includes
fix stereobm uniqueness check
just playing again remove the space again
uncommented fix of write read keypoints
sema weaken a validatedeclfornamelookup to a validateaccesscontrol in computedefaultaccesslevel
rle dse refactor rle nfc
fix case with several same keys in join
bug fix backup snapshots could take unexpectedly long if the desired snapshot interval is less than the configured snapshotdispatch interval
fix compilation error
clang format in http config plugin
moduleinterface add printingdiagnosticconsumer to sub instance
update mergetreedata cpp
bug fix clarified a comment
constraint graph make sure we print type variables when dumping components helps with debugging the constraint graph
update avrorowinputformat cpp
report an error when the impossible tm happens
wrong access to ccapplication sharedapplication pointer
build fix take two truly disambiguate pow
folllwup to planner splitting
change the end of sd cart print message from minutes seconds to hours minutes
fix fix build of afp for linux for openelec
checkforextradatastores fix compilation error
fix tag needs to be assigned directly before using it
src main cpp endian compatibility in packet checksum check
remove exception and add some documentation to function
time sometimes moves backwards coming up from base libraries
fix widows tests
make dbtest shutdown cleanly
remove a direct settype on an expr in the constraint system noticed by inspection
fix deleting from without rowid tables
remove floor so that global pos works with scale
invoke codeformat on inbox changes
don t check return value of doinitialization it s always failing
fix error message
remove imports of silmodule h from files that don t need it no functionality change
fix for gcc
add listen to gui_msg_update in guiwindowfilemanager
fix thread race condition on shutdown
fix a crash found by nightly correctness
this time fix the leak properly
sort labels in right click menu
handle edge case
explicitly check for clang and g
disable javajstest in the right place
omxplayer always hide seek dialog when player signals display after seek
always do commitifneeded during delete
fix unimpl unity methods
use subcommandargs instead of argv within subcommand if block
utils check style check style n
fix small bug in recover_reversible_blocks
support reshape with zero dim
fix compilation error in release mode
adapt to upstream llvm api changes
pvr pvrguiinfo fix cpvrguiinfo get starttime duration to fetch data on demandinstead of using outdated cached values
no need to repeat namespace
fix oom error message in tensor constructor
remove unnecessary setclipregion around multiimage rendering guitexture clips when it needs to anyway
fix compilation tag not being set for albums
ios remove clamp to one cpu for ios
clear retracted status when homing the z axis
merge branch master of github com haoel leetcode
dvdplayer increase interval for calc of average audio offset
showing pattern matches if a region was identified
fix crash in unloaded userpic handling
fix saving of option int slider add on settings
don t add virtual column to empty block
fix trackerbase close children completion check now verifies that all children have completed instead of only the last
update storagesystemparts cpp
fix mac os build
selectable text baseline alignment for line that aren t of text height
merge remote tracking branch origin master into master next
fix veryify in createextent
squash this style fixes
pvr pvrguiinfo use operator instead of memcpy every aspect of memcpy is unsafe
fix missing include
fix memory leak
corrected bug and leak in getbool
fix truncate for geo
fix function signature in cpuprofile cpp
add lock holder in all lock taken paths
fixes crash after remove_line in richtextlabel itemframe always have a line
fix assert hit in eventmanager due to queueing system
dialog try to unbreak windows builds
merge remote tracking branch origin master into master next
remove conditional compilation regarding jpeg and png support it s just always going to be required for imglab
only show timing information when clock is used
browser error msg
test cases all succeed again
be tolerant in remove with isrestore true and no rev given
get correct control code
itemadd re ordering the tests submission should now be faster
try to pacify some compilers
improved cudnn error messages
printasobjc std function llvm function_ref for some non escaping params
fix wrong condition
csranking delete a pointless old ranking hack nfc
fix return type is not right
fix create albumview with mysql missing comma
make pin icon bigger in popupwindowpin
sourcekit remove unused variable to silence warning
fix cosmetic c p error
add mpd to video extensions
roll back _ycoordflipped changes
more diagnostic fields
fix mirrored effect3doutline
ast use new module shadowing rule for qualified lookups too
fix rpc call name in error message
fix comment type from my previous commit char not char
walletdb fix syntax error in key parser
fix a typo in transactiondesc
dvdplayer fix waiting time for buffer when ff
fix a use after free error the code was trying to access a debug scope of the erased instruction
gltfio clean up finalize
show user profile from remove users
fix infinite connection loop
fix compile error
merge remote tracking branch origin master into master next
fix typo explictly explicitly
fix retinademo compile errors
softfloat f64_sqrt bug fix
fix code style
ast fix genericsignature requiresclass for layout we forgot to check for a layout constraint here
minor user output change
fix broken compile
fix test for stacked paths before checking if item is in rar zip
dsp remove useless spam log for signalinterrupt
the keyboard xml mod attribute doesn t distinguish between left and right alt
fix extern type
remove unused param
release group memory faster
do not hard code a result type that the type checker will fill in small clean up
addition to prev revision
fix transaction no commit
constraintsystem prefer disjunctions of bindings over other disjunctions
remove an ugly line break in a transaction tooltip for case transactionstatus mature
fix null pointer problem
fix shape of randomsample operation
use default handler for sigquit
tweak extruder errors
tweak the import compression property hint for clarity
rendermanager hookup hw dec renderervtb
kc is also a multi pid parameter
fix perf tests
invalid return value for guiwindow initialize
api versioning is unnecessary for the metadataversionkey
try more than the first address for a dns addnode
tbdgen only normal conformances get symbols
activeae align vizbuffer size with sink s max delay
add new feature show databases like
silence eq band c1 c2 and c3 may be used uninitialized warnings
update maxsubarraysum cpp
activeae make activeae default audio engine to get it tested
fix regular refresh of rss feeds
be sure to set correct pip maximum size
adjust branding in datadir lock error message
update pr comments
update mergetreedatawriter cpp
cleanup nextafter workaround
made unit test less likely to false alarm
reverting testcpp appdelegate cpp to use config file but modify it to a suitable value
fix left shift warning
fix a comment in net cpp
fix some memleaks
fix exception when roi for generated sample is evaluated out of image borders
query time count is a ulong not a long
remove already_signal_handled guard because it is not actual
mouse fix mouse motion pausing game on linux
squelch signed unsigned comparison warning
tests add fuzzer initialization hold eccverifyhandle
increase epsilon for mog2 getbackgroundimage test it uses float arithmetics
uncomment the code
ctexturecachejob add fallback hash in case neither mtime ctime nor size of an image are known
ensure that statistics blocks on freelist have _released true
android drop shield workarounds not needed anymore
improve test coverage for service tcp udp
fix unit test failure
do not delete response some other method will do this
update zookeeperimpl cpp
upnp use a reference to m_streamdetails rather than taking a copy
handle an unsigned comparison with a constant
shell logging when starting stopping mongo programs minor
remove unused includsion in thrift client
examples clang warning fix
activeae make sure getdalay returns no negative values
correct treatment of erase character with repeat
framework fix camera dag path
fix bug on outputfilemap parser when parsing beyond json
fix variable names change undeclared identifiers running_ and paramsqueue_ to their proper names
use std move
fix comment broken in last commit
irgen fix a warning
colorbar change default fg bg colors to white and black rgb
remove features that aren t supported in anim format
fix invalid fanart appearing in addon views
made copying to matlab a little bit faster
update replicatedmergetreeblockoutputstream cpp
httpheader parse fix parsing multiline parameter used only in tests
wrap dumpwallet warning and note scripts aren t dumped
fix append qb extension to incomplete files feature for torrents add from magnet links
fix cpipefile write return value the internal write return bool we need int
more namespace db comparisons
bugfix use testnet requirestandard for acceptnonstdtxn default
made gcc happy
use fetched value
oss addpackets must return int_max on failure
since we are using smart pointers an explicit destructor is not required
add missing include functional
fix a crash on multiline plates when no line is found
fix script and visibility icons s highlight height in tree
fix predefined sorting of musicvideo nodes when accessed through the music library
fix inconsistent whitespace
rid cached class was wrong mono
its nullptr on windows too
fix comment on max trx decompressed size
settings fix default values of locale language and locale country
bug extra chars
driver fix typo half sentence and period
osx yield to main loop after input action has been handled
editdialog set focus to text edit after clearing
fix bad merge
fix broken build invalid use of qualified name std advance
fix asan in cache test
fix typos in ccrendertexture cpp commnets qualcomm andreno gpu adreno gpu
renamed session to handle
constraintsystem switch to usedc for getunopenedtypeofreference while calling from gettypeofreference
pageview fix that the indicator s direction is not appropriate when pageview setdirection is called before pageview setindicatorenabled true
fix shader crash if duplicated struct members created
epg refactor align m_ilastepgcleanup naming with actual usage
remove unnecessary empty line from dictionaries
guilib remove double call to updatebuttons
fixes collisionshape extends not updating when draggind handles
remove prewarp for detection scan area already handled by calculating the area on every resize
compiles on windows
simplify and fix logic
fix typo in fix fps help line
activeae set encoded rate when doing transcode
fix jni binding
made session deletion a bit faster fix possible memory leak when folder scanning is enabled
revert change for storeddiagnosticinfos
stop and wakeup screen saver when play request received from any source including playfile
fix minimap drag when height is less then control size
nfc fix typo in serializedmoduleloader cpp
fix the string get_base_dir logic to properly check for top level directories on windows
wificlientsecure don t trash certificate and private key on stop
add an additional nullptr check
ae fix normalize volume on reconfigure
merge remote tracking branch origin master into master next
fix applied_block call
if redirected toa udp stream re create the input stream
ensure dvd is picked up in isdvd
skins process startup xml on skin change
irgen remove heap allocation of std thread
imgproc fix using of uninitialized edge members in fillconvexpoly
applied a patch from nicolae vartolomei
fix skew settings report precision
fix assignment of class labels by using the indices from the sorted distance vector of pairs
use thputils_assert rather than therror in torch csrc module
release pressed action if event is remove
add handlers for remove update save replace
remove redundant predicate
remove false positive warning
merge branch master of https github com tencent ncnn
fix bug in palette toallegro index out of range
nogpu cpp now includes bestgpu h to see the cpuonly define
launcher fix a crash in browsefileopen
fix compilation warning
update aggregatefunctionmovingsum cpp
cosmetic cleanup view creation in musicdb
comment the last three test cases
fix warning during cross compile for arm
disable addmultisigaddress if not testnet
remove type check in gpu pyrlkopticalflow not it support all depths
if applied this commit will describe permission denied message in imwrite_ function when user does not have write permission
followup patch for tmc stepper init
preserve action name in capplication processmouse
use istriviallycopyable instead of std is_trivially_copyable to work around issues on linux
add minor test
fix error codes
startup log fixes potential problems
remove nulltype from assignment
fix calibration screen showed full screen rather than windowed
print rescan timing info in debug log
eliminate unnecessary check
gardening replace dyn_cast x y with isa x y where appropriate
fix setcolorselector command label in keyboard preferences
experimentalhtkmlfreader add checkdata option for filtering seq of uneven length
lua rename colormode grayscale to colormode gray
async gpu only do reactive flushing on extreme level
softaesound remove unused struct
remove unused define
fix typo in docs
typechecker call basegenericctx isgeneric before calling getgenericparams otherwise it can lead to a crash
fix help message for rpc getpeerinfo
vcpkg hotfix for non windows don t include versionhelpers h
hide tooltip if new mouse position s tooltip is empty
probably should log it on this case
fix skew matrix bug
remove outdated todo comment
parse remove unused findfirstnonnull function
cxx restedgeshandler now actually returns a result all tests are green now
pvr workaround a deadlock if no pvr addons exist and you activate the pvr manager
if mutableconfiguration exists skip checking rawconfiguration in databaseconfiguration
cssimplify handle coercion to wrong type for function result locator
simplify the code simplify the code as speclad suggested
only reload current image when not switching to next and current is not video
do not return false result
removes unused enum in performacecallbacktest
fix fix a bad trace log statement
add mutex header needed for some platforms
rendererdrmprime align render area
verify yieldinst values against substituted function types
cprogressjob use markfinished instead of manually calling cprogressdialog close
fix plr condition saving retract
kill warning from unavoidable signed unsigned comparison in bitcoinrpc cpp
iso prep more ifdefs for apple arm
merge branch master of git github com mongodb mongo
fix listunspent rpc convert parameter
initialize all instance variables
dropping sharding prototype collection more verbose
fix importing gltf meshes as external files
videodatabase speed up the video database cleanup for archives
mark the first shape as inside not the second shape when csg shapes are co planer
weatherproviderlogo property cleanup
update mergetreeindexconditionbloomfilter cpp
add randaddseedperfmon to makenewkey
move a signal outside lock
transferlistwidget keep columns width even if they are hidden on qbittorrent startup unless something goes wrong
remove commented lines
fix make build
fix caddon hassettings was broken after a call to caddon savesettings on creation of user settings
fix js path
skip raw moves to current position
fix return code of mtime
frontend use the private module interface when available
delete duplicate log message
add missing lock cs_main
fix bsonobj valid
cosmetic fix comment
add use glprogram before setting uniform data for meshcommand
torrentcontentmodel code clean up
actually return the value
fix layout in intro settings
fix test compile
don t copy lockmove flag in document copylayercontent for a second background
addons log execution time of caddondatabase getaddons
add missing include in testfilefactory
fix problem where the script lists gets ordered even when ordering is disabled
fix compile warning
more aggressively cleanup a bad process id file in simulation
javajs cpp if the appsrvpath is set on command line respect it
lstmtester add missing space in log messages signed off by stefan weil sw weilnetz de
make threadopenaddedconnections2 exit quicker if getnameproxy
cddandroidandroidjavaengine cpp with updated method declaration
io cpp changed back cv_load_image_color for earlier versions of opencv
suppress compile warnings
add read lock for correctness when determining whether a rollback is required consider moving this to an atomic high watermark
check the incoming message length before allocating memory for the message
allow to rename animation just after it was duplicated show animation name in delete prompt
device handle should not be a random id instead it s the current npad id found during hardware testing
update executequery cpp
cguiwindow resolve includes on copy of xml element to avoid manipulation of stored xml element
jni fix local ref leak
fix dialogs in projectmanage don t with editor theme issue
merge remote tracking branch origin master into master next
gui reset view state directly after directory was fetched
add performance test for cv_32fc1
remove unused message
sema assert when asked to infer an associated type for an imported conformance
osx fix build missing include
merge remote tracking branch origin master into master next
fix a deserialisation bug for aqlitemblocks
can t scroll with mouse wheel when window is collapsed
fix merge distributed with join
turn off rocksdb in simulation
updated the adjustcalendardate to account for the transition year quirk in the japanese calendar
type checker minor crash avoidance tweaks
fix uninitialized variable idcontent in videolibrary cpp
add dns seed of open nodes org
minor main cpp changes
update test utils for windows
fix for deferred_transaction_tests
add a new example to check correctness of the code
fix build with capnp
remove previous use of queryresult json in js_executeaql
moved output messages to log handler
add missing lock in cnode copystats
fix crash via division by zero assertion
update tracebrowser cpp
gui dont attempt animating steps when not debugging
improve utf8 encode coverage via writing to autoutf
fix crash when interf was not created
no need to check ultipanel with doglcd
rpc submitblock returns null on success string on error
refactored pagechecksumcodec checksum
modify bundlereader and the version read
tweak the name
disable flaky tests in testextfile and testextimagesprite
simplified example program
cleanup videoplayer remove unused variable
add test checking we don t throw when creating genericindex with lshindexparams
xml reporter reports warn message when not used with s
update catboostmodel cpp do not evaluate catboost model on empty dataset
renderer fix hasframe
change trace event message if heap profiler is not running we can t take a snapshot of the profile
fix onnx deconvolution
fix crash in update checker destruction
fix potential crash when an empty url element is returned while scanning
fix get_shader_type index upper bound
waitsynchronizationn handle case where handle_count is invalid
fix few warnings on linux
fix alignment of coloredit4 label
made comment more clear
pvr delete epg tags on destruct
allow to open multiple projects when some are not imported or without main scene
fix a debug only silgen crash
quiet msvc warning
activeae initialize engine stats before first usage credits to jimcarroll and valgrind
silprinter be more tolerant when printing a sildeclref with a decl without name
properly implement idisposable
gardening fix comment nfc
integrate fix quotes in command in case of space in the path
change a test image of pyrlk
missed one x_enable_pin
fix crash using import sprite sheet active tool nullptr
we cannot buggify replace content bytes because it takes too long to recovery when the txnstatestore is too large
fix issue in boxfilter with destination depth not equal to source depth
check valid configurations of variable flags
remove whitespace changes
fix accidental stop of voice recording in sections
try fix header for ttlblockinputstream
cosmetic remove commented code
animationplayer fix scrubbing after play backwards
environment get rid of compiler warnings
indexindexintoab is only vald for delta
fix possible crash when if is down and airplay tries to fetch the mac adr
dbhub do no show error message when download was cancelled
fix some warnings in timeline cpp
remove when hphpc prints the hhvm command line
recreate externalauthenticators ldap on config update
remove warning look at return code minor
don t show contextbar when the program starts
fix crash when setting material s next pass to itself
dropped some garbage lines
opengl keep stencil test and framebuffer depth_format in sync
improved execute method
imgproc hdr fix bounds check in hdrdecoder checksignature
trival out of index fix
checking erasable concept
improve the code by code review suggestions
add isvisiblebyvisitingcamera check when visit
freebsd add missing includes in xrandr may be it s not the right place to do that
improved error message on missing optional config
add missing setintroducedin tag
fix addimageasync bad judgment of generate image for the first imagefile
fix crash if we close a tab when we re dragging it
update actionsvisitor cpp
allow zero raise for deploy stow
convert test back to legacy producer keys
gui fix more bugs with updatechecker
diagnosticverifier fix asan issue where twine was stored to a temporary
update tabseparatedrowinputformat cpp
update context cpp
fix rounding bug
videoplayer amlogic bugfix for instanceguard locked
add const to parse param
prolong resume data save interval
fix armatureanimation setspeedscale
audiocodecffmpeg disable drc in audio codec context when advancedsetting applydrc is disabled
allow all file names when opening a db by drag drop
canny disallow broken inplace arguments
fix new call to lookupdocument
merge remote tracking branch origin master into master next
cnoderef copy constructor and assignment operator
better serverstatus logging
implement frametags add remove member functions
limit remote timeouts to max_execution_time for selectstreamfactory
mdns fix some possible crashes seen when using embedded mdns
made test more robust
statusparagraph minor refactor
cleanup videoplayer silence sign compare warning
tapi enable some disabled tests
add check in storageimplinforequest evaluate instead
merge remote tracking branch origin master into master next
fix a typo
merge remote tracking branch origin master into master next
charsetconverter do not skip last zero if original string have it at end
buildbot all sharding tests remove unnecessary namespace check for checkshardingindex
cli create getnewaddress
tests disable gettemperature test for testcpuinfo on windows
shader image import predcondition
cddb fix broken hello command should be as cddb hello username hostname clientname version
fastrestore send reply before assert fail
lib ide improve error message outputted from getplistentry function nfc
rangeinfo simplify some code by using stl extra nfc
move project manager favorite icon back to center
fix bad solution
use any type for bool value
fix signed unsigned warning error
add default focus stylebox to menubutton
update forkprocessor cpp got rid of an unnecessary local
typechecker fix a crash
gardening fix formatting of recently introduced file header
osx silence kaugrapherr_nodenotfound error it is harmless
reorganized rss feeds popup menu
torrentcontentmodel fix crash on exit
after files relocate don t remove the old folder even if it is empty
use checks exit_with_message
fix filter input in rtl locales
fix compile warning
add restartingthread wakeup for faster session recovery
adjust an expression i forgot from earlier code review
separator remove unnecessary computation comment
fix unit test compile
shader language add in for declared variables within for block scope
fix lattice index for last sequence
add length info when serialize stringbuffer to string
little translation for hungarian
remove unused variable
add handlesimpleerror for database not found
gui remove initial zero on the hexeditdialog
cdvddemuxpvrclient zero init props in requeststreams
log the difference between process rss and memorytracking metric
enabled networking tests for winrt
update client cpp
make palettepopup hot region a little bigger
dbg use unsafe memory reading in thread cpp
fix easy error
finally fix selection of destination of move ttl
merge remote tracking branch origin master into master next
fix sharding key is not deterministic message
fix typos in comments for examples
add wait time after dd not found error
add stored property default arg symbols
add filename to the exception
change stringbuffer to memorybuffer in discovery sdk
input inertialscrollinghandler remove unused variable lastpanpoint
constraintsystem cleanup remove else after return
redraw the whole screen after fullscreenpreview
videoplayer vaapi temp disabled runtime check
dyncom increment addr when accessing lr in ldm
add missing studio change for director drawscene
default item width for alwaysautoresize windows expressed as a factor of font height
closed state has to be set before quit websocket thread
remove empty block
fix wrong debug check
fast path for named property access
fix possible memory leaks in iso9660 cpp
log if flushing is taking really really long
net add seed bitcoin sprovoost nl to dns seeds
change effect file
small chunk fix minor
fix traincascade read issue
add all flag to vswhere exe call
fix the out of range crash in ccarray randomobject
modernize pvr range based loops
fix the path to the testdata
improve the perfermance of clear background for camera
merge remote tracking branch origin master into master next
re add the openssl hack was never mean t to be remove without add a proper fix
cosmetic misplaced parenthesis
aml fix sysfs path for permissions check
merge remote tracking branch origin master into master next
fix null termination of wsave_path
fix utest fail
dbg fix a bug with detaching while paused sometimes
merge branch master of github com yandex clickhouse
simplify the code
grammar forgot to update the ast treewalker too
remove unused variable
csettingaddon remove unused define
fix test enough size for detection matrix
better error message
gardening fix recently introduced typo
peripheral add peripheralimon to peripherals cpp
appmessenger send announcements upon dpms state change
qt show transaction id in details
once we deleted a query from the registry we don t need to close it anymore
add a warning when jit falls back to autogradclosure
fix multisink example
merge remote tracking branch origin master into master next
art cleanup no need for convoluted resetting of thumb now that it s just a fallback
more reliable temperature serial report
cubemap fix duplicate flags property
only enable nscoding related runtime name warnings with objective c interop
making sure motion rect is in bounds
add workaround of old gstreamer
update expressionanalyzer cpp
gardening use consistent variable naming
fix dialogmutebug not showing when volume set to minimum
fix a bug with openvino backend
fix the fix
cguiviewstatewindowpictures cleanup getextensions
change wording of a message box
delete default skin font on exit
merge branch master of https git codeplex com cntk into linux gcc
correct a caching bug in clusterinfo
net add dnsseed emzy de to dns seeds
apply datadecimationfactor to cross validaion by shrinking minibatch not decimating it
extend huge local cache map crash annotations
fix problem with preview not showing editing entry values on filters
remove unnecessary and incorrect std move which makes some compilers unhappy
fix wrong sort order list for music playlists node
substitutionmap make conformance lookup robust against ill formed conformances
make folders clicked in the filesystem dock open the selected folder
fix to servo allowing write to be called before attach
fix build after merge
add more comments
fix odbc sqltype mapping
game osd don t display seek dialog after gameplay resumes
add clearer output message
gui fix a bug with signed unsigned values in the wordeditdialog
add ability to go to top of media panel by clicking on selected tab
check offset in exifreader getstring
compile w out readline shell
fix crash in shared media save state
cosmetic remove unneeded var
update dictionarysourcefactory cpp
make warning about reference singletons more accurate
fix bug with numeric queue ids
ast don t try to get member substitutions for a non member context
smb make sure we deinit samba before global destructor library can have been unloaded by that point
inputmap rename joypad button ps x to ps cross
update externalloaderconfigrepository cpp
minor change in keyitem restorekeys logic
make checkdiskspace portable
remove unnecessary copy of ccbmfontdef structure in cclabelbmfont createfontchars
win hook up logger of threading library at app start
fix status text after reindex
ae fix number of provided buffers for pt
avoid to start with both colors selected fg bg at the start in the color bar
add serialization unit tests for extreme integer values
videoplayer fix some error handling
update lights when their reverse cull face mode is toggled
in qtorrenthandle cpp exception made caught by reference
merge remote tracking branch origin master into master next
fix intermittent cmgen crash
mysql table names should be case sensitive
fix no json file bug
add cfg tests
revert back to join before stop
dialogvideoinfo remove obsolete path invalidation code
merge branch master of https github com microsoft vcpkg into kinddragon fmt
fix couple of possible crashes
test use fs namespace in dbwrapper unicodepath test
update topological sort cpp
lstm replace null by nullptr signed off by stefan weil sw weilnetz de
fix shard numbers output in logs full cluster had been printed over optimized
remvoe hstrs warning
timeline fix bug start cels movement with right click
don t ask client to send write info byte if the version cannot handle it
mangling handle new mangling suffix in type reconstruction
fix slice point overread
don t use timestamp for other things since it may change
fix readline ambiguous completion on ubuntu
buttonbehavior default assignment of pressed behavior now also test for imguibuttonflags_pressedondragdrophold which wasn t the case before
fix conditional expression constant warning
core set is_powered_on before gpu is initialized
remove some todo notes remove two todo notes as they have been dealt with by now
print correct minimum mempool size in mb
another compilation fix
add missing braces
pvr fix recordings unwatched overlay
dvdvideocodecdrmprime enable use of threads
dvdplayer set default screen region for dvb subs
don t throw unhandled exception if subscriber fails
increase severity to sevwarnalways for toomanystatusrequests trace co authored by tclinken trevorclinkenbeard gmail com
fix mac build
prevent readline history memleak
fix bug in fast
fix another wfloat equal warning
dbg speed up with waiting for threads to close
primary index does not need to free anything
fix linux build
check original title during movie search
fix typo in comment
fix region_is_masked function to correctly respond if region is masked
fix crash when waking more threads than are waiting on an address
api digester use target triple to check platform info
fixups mistake auto commit
fix msan report in querylog
fix fix wrong definition check for xbmc getcondvisibility system platform osx
fix busy wait
rename current_block_time to pending_start_block_time which is a better description
bugfix wrong _outputbuffer clear
sil fix an assertion
try curlfile seek without multisession on failure
cec extra guards around m_querythread
fix typo incomming incoming
add gettotalstepcount stub
addons set addon installed if m_localaddon is available not if m_addon is available as that s addons in a repository
fix time returning epoch time in milliseconds rather than in seconds
make buildinfo command admin only
eden video art all season art wasn t updated from eden frodo
blind commit to fix compile warning
gpu fix a missing format specifier
slightly relax batch norm check
updated winrt include
selectable menuitem don t use frame rounding for hovering selection
fix warnings under windows
old bug fix transaction reset not being called in a retry loop
add a comment telling the user that there are higher level threading tools they would probably prefer to use
chg cec assign the setup menu button to title i e context menu
android call codec release
ios fix lost calibration settings
add support for tif bmp images
ocl update program cache defaults
fix align center info is not saved with checkbox checkbutton
fix compiler warning
disallow creating new projects in a non empty folder
better grammar in comments ci rerun
change default onionskin opacity step setting
fix infinite loop
fix performance improvements w r t headers
nfc add a typebase conversion constructor to nevernulltype
update hog cpp
fix how limits applied to artist and album smart playlists or custom nodes
gardening can t can t
fix variable shadowing
fix bugs of page view
fix crash in formatopcodestring
jni fix crashes when finding external storage regression from jni merge
charsetconverter finally remove cstdstring
fix stagefright do not exit decoding thread in case of potentially recoverable error
disable ipp integral in ippicv version
fix assert numbers from separate patch
fonttest was not rendering ttfs
tweak marlinserial precursor
called after setdevice wrap in a cudasafecall
pretend to use result marked as warn unused
add noexcept to customstdexception destructor
integergreaterthan use getint saves int string int conversion
update summingsortedblockinputstream cpp
driver tbd files are linker inputs
check that we re calling the widget invalidate from the ui thread
update weak hash
fix another memory leak
use linear feedrate in gcode_t not volumetric
fix normal params in getdisplayresolution
subtitles fix smi subtitles to use quoted start tags
add default pretty printing for console mode
improve locking in sqlitetablemodel
fix graphics fillareabetweenrects
rocksdb returns certain statistics values as strings honor that
remove conditions about ji_initialized in skinneabletheme init_widget
rle dse improve how rle removes instructions with no use existing tests ensure correctness
remove unnecessary temporary variables
ccoinsviewerrorcatcher raison d etre
add logging whenever a wait failure client triggers
don t suggest emoji on non letter
expose itemlist unselect_all
check connect ports with same header
explicitly set opencv functions to single threading
fix binary in place editable cell
fix return proper error
now the more options visible status is saved in configuration file
expose params in python interface
fix combine buffer
set default value for _lightdir
services hid add activatenpadwithrevision to the hid function info array updated based off the information on switch brew
combine studio change for get program file full path
pause audiostreamplayer on scenetree pause
keep editor description metadata when changing a node s type also copy edit group lock when replacing a node3d
addonmanager fix don t crash on too early calls
fix csharpinstance set not working with base classes
update sendfrom rpc help to correct coin selection misconception
add todo for sheet type
fix typo wit with
merge branch master into master
fix possible infinite loop in qbtsession constructor when web ui is enabled
prefer result from inverted image if the mean confidence is better signed off by stefan weil sw weilnetz de
fix for linux
as pointed out by blue marlin the previous code would raise the z axis twice for a servo based probe
add sw options for openmp
update vntapmd cpp
fastrestore fix corrupted trace event in fastrestoreversionbatchprogressstate
clean up logging of outbound connection type
fix system time hh mm ss et al
remove localtime from httpdictionarysource getupdatefieldanddate
update macros cpp
add a recursive solution
prepare for soa
yuzu fix title bar display previously the version number got hidden after starting a game
print objects of specific classes better in jit type tostring override unit failures right now it returns
fix default layout dock sizes
update fdbserver logrouter actor cpp
fix wrong equation in accuracy break rule
improve error message
remove weakened test cases
widen prediction interval
stats remove tracer on lookup direct too noisy
use value stored in count value of collection when re loading
tables fix lower clipping when using imguitableflags_nohostextendy
mono fix bindings for parameters in vararg methods
update expressionanalyzer cpp
batchmode remove a low value quadratic assertion
gardening remove dead code in serializedoc cpp
use float for cv_pi
don t advertise on irc if we don t allow external connections
update dictionarysourcefactory cpp
constify argument to findnominalsandoperators
register wallet endpoint
include qdatastream header to specify the operator
cleanup cguidialogvolumebar onmessage
kernel thread corrected a typo that caused the affinity mask to never be changed
fix capabilities report
fix negative light flickering
fix typo in nativescript property getter
method list no longer forwards drag commands to script list
better error message for malformed plugins
merge branch aql jmmh conditions of https github com arangodb arangodb into aql jmmh conditions
add blocksonly mode
wallet replace boost none with nullopt
fix layout menu labels and shortcuts
settings only register the cwakeonaccess settings handler once
remove accidental semantic change to curop ns from before
fix own world option of viewport it happens when the viewport has worldenviroment child with a valid enviroment
chunktests reorder includes so that pch is first
dbg acquire ownership of message stack before freeing it
allow key ids in keyboard xml to be in any base
no need to start keosd for cleos multisig review
fix error when dragging empty script list in script editor
fix test to use numeric_limits instead of infinity
fix invalid mono log file name on windows
add a couple of sanity checks to make sure the datum buffer sizes matched what we expected
add a comment
really fix tests
fix segfault on postprocess sort
don t ignore fs rename errors
calculating thresholds before char analysis for classifychars
add accept_event calls for scenetreedock and basebutton
add bold font
fix request pending webpage
fix the build issue in osx
trivial optimization use getancestor to compute new target
update mongodbdictionarysource cpp
add detections vector initialization in tests
fix regex for s3 uri to pass clang tidy check
minor array js round trip test
add support for vertical scrolling with mouse wheel in paletteview widget
mmalcodec move m_output_busy into lock
rpc getaddressinfo rpcresult fixup
gframe as internal node in gexecutor
fix suppress expected errors in data distribution
music format album rating
the mini editor cannot be the current editor
merge branch master of git github com mongodb mongo
swapped x y in elbp_
storagekafka use config string instead of just server name
irgen change startoffset to startoffset nfc
updated ccfileutils of linux
cut down on output when mongodb fails
fix method name
fix settings show animation
merge branch master of github com mongodb mongo
add expiration and pending block time to expiration assert
check invalid usage of drop_range_op function
merge remote branch refs remotes svn trunk into svn
irgen document a bool parameter at the call site nfc
don t crach when opengl version is unsupported
fix gcc warning
dvdplayer fix calc for frame dropping when going ff
pageviewindicator fix that indexes were upside down in vertical view
fix build on linux after bluetooth commit
controller dialog update controller list after controller add on installation
fix winrt macro
remove unused include
fix possible null pointer dereference in dvddemuxpvrclient cpp
fix gcc warning wreorder
demo tweak irritating pink color
explicitly cast thread id around the bitwise not for platforms which don t implicitly cast pthread_t to int
update compressionfactory cpp
bug fix restore start dry run would not open a trace file
don t double delete a pointer m_searchfilteraction is owned by qt so we shouldn t delete it manually
constraint solver don t mark a moved from instance as diagnosed
fix for cmp16s condition during ipp compare type convert
avoid referencing a dead workspacetabs from m_droppreviewtabs
remove the duplicated http status line when error
databaseutils remove duplicate handling of fieldsorttitle for movies in getfieldindex
should finally fix permissions issues on windows
this variable is accessed by multiple threads with no locking changing it to use atomic ops
pvr json rpc fix pvr addtimer for timer rules
fix several metatype to objects that didn t use cleanups
one more test for read through boundary
pvr fix playing items from library
prefer to disconnect peers in favor of whitelisted peers
fix bad conflict resolution
fix musicdatabase compile warning
prevent unlikely division by zero
important http proxy fix
fix shutdown airplay and airtunes server on app exit
small change to check_entry_behavior test comment
aml xbmcapp drop aml workarounds
thread check that thread is actually in wait state when verifying wait
fix tab selection after closing a reordered tab
csettings now we can savely call clear again from the destructor
merge branch master of https github com telegramdesktop tdesktop
fix my dumb test
renamed a function so auto complete in vim works more conveniently
use logical or minor
sil fix ownership of getcurrentasynctask builtin
fix revision id bug
lambda fix missed in review minor change for portability gcc is technically violating the standard compiling this lambda as is
fix missing texture download error with opengl es
gui fix a bug with the double click signal in stdtable
use ui where possible
fix get rid of unknown nodetype requested warnings when browsing shows node in video library
fix document userbuffer test
clean up logging
introduce an astcontext local variable in the method
give hard coded seed nodes a random last seen time to randomize order they re tried
fix invalid access to outdated iterator
fix compile errors
contributors missing in project manager title
fix group call bar in legacy groups
update num threads in processors
test faster less
comments and clarifications
offsetof only supports pod types no c classes
work around visual studio bug inferring type of auto
fix g_curlinterface easy_getinfo with curlinfo_content_type asks for a char not a char
memory return false for large vaddr in isvalidvirtualaddress
fix unused local variable warning
make repl worker threads have a unique name
fix flow matrix dimensionality in emd
add a few more blas binding tests
fix code review comments
gardening fix recently introduced typo is it
check for active replicas when waiting for commands
add some debug
fix bug of log message
android fix drain in flushed state
set return value
fix error in last commit
fix possible ambiguous condition on render renderlayer
fix clang tidy
lua wrong callback setting for webview setondidfailloading
update interpreterselectquery cpp
document both the peerbloomfilters and enforcenodebloom options
deleted some unused codes
sema fix closureeffects to pass the right declcontext for the lookup
label fix letter spacing may be wrong
fix memory leak in new properties panel
examine return value of fail point to placate clang
fix droid keys handle f keys
add handling for std alloc
earlier change had a sideeffect reverted
fix password prompt
merge remote tracking branch origin master into master next
cosmetic in cguiwindowvideonav
fix accessenforcementopts in debug builds
fix undefined behaviour in traversercache
activeae do not set fading flag on a passthrough stream
remove headers vscode auto add
initialize dispatchers after another process check
don t re use cursors
remove new lines
sema remove redundant call to validateaccesscontrol
linker changes final
amagad fix fuzzing
fixup broken build when enabling amlplayer
peripherals fix locking wrong critical section
fix test case
hide mark as read button in notifications when app is pass code locked
fix progress dialog s bars overflowing
fix run time error
core reset var to null
invalidate guibuttoncontrol when label is set
texturemanager fix possible invalid pointer deletion
correct the wrong boolean statement and make it easy to read
add back key support
remove extra newline in help and set reuse address
describe reasons when query profiler cannot work
remove unused code
arangoexport fix csv export escaping
fix clean database will fail on mysql
fix crash generating thumbnails for certain sprites w cels in specific positions
mark connection done when finished with it
osx fix dvd couldn t be played when disc was inserted after xbmc was started
light respect editor_only setting in release build and dont show the light
translate this file
capture cocoa mouse move
fix registry test
fix possible crash in player closing
fix update rating query typo
adjust scanwriteunescapedstring test case
merge remote tracking branch origin master into master next
fix text search bar not correctly switching from replacing to just searching
touch input fix for touch screen input in linux to see presses as mouse clicks otherwise nothing can be selected
fix uninitialized memory error
daemon mode add sleep loop to prevent defunct child process call setsid in child
clientbasic for client
fastrestore transform keys when db is locked
pvr recordings window fix duplicate entries in recordings listing
merge remote tracking branch origin master into master next
freebsd include stdlib h for realloc free
remove an unused and leaked checkbutton instance in theme editor plugin
fix available time for garbage collection to be not bigger as the callers value of availabletime
dvdvideocodecdrmprime set video picture chroma_position and colorbits
remove dirname variable
fix log spam when the temp fonts directory does not exist users who never played an mkv with ass subs
lm assert on unsupported multi message
network fix some wrong upnp related calls to be able to stop the client and server
editor run validation on every text change not only inserts
gdscript allow new to be called in non static functions
fix wrong unread counter in history top bar
small fix in finished list
sema small cleanup in csapply
fix amc bail out on null hint size
break out of scan loop if can t find adapter on first run
allow change slideshow direction
don t disable the browse table option when browsing a read only database
pvr cpvrepginfotag fix first aired value respect that this attribute is optional
changed the text from cancel to done in the fix dependencies dialog
fix render release bug
no lock for signal better startup of new threads
remove unknown fields from gx_cmdbufferheader
changed improve handling of internet filesystems http https on lans
modified minimum scroll bar thumb size
updated output on prepcharsfortraining
align dlie and ngraph average pooling layers
fixes navigation path reset
initialize logger at the beginning of program to avoid threading issues
fix logged number of inserted rows into replicatedmergetree
fix peripherals settings dialog not destroyed on exit
fix inspector update after changing anchor via buttons
catch ios_base failure specifically
code clean up
pvr support epg item type for listitem premiered firstaired gui label
simplified fix for babystep pulse width alternative to the apparently superfluous double delay
change get_info priority to medium since it is used to determine status of node as it is syncing
delay the call timing of circuit reset and wait until the health check is passed
sgemm is always faster on common channel size
initialize variable to avoid windows assertion error
update eosio token cpp
add the necessary comment
thread sleep doesn t give any upper bound guarantee test failed sporadicaly
fix uninitialised value
update settings cpp
increase backup worker memory buffer size in simulation lower limit causes backup worker to become stuck
fix a bug with full allocation mode introduced yesterday
pvr pvrthumbloader cpp fix include order
merge branch master of git github com mongodb mongo
aesinkpulse if we cannot get the buffer from stream we have a problem
missing white space in function arg
fix race condition for cascade classifier when tbb enabled
fix possible null pointer dereferences in guiwindowpvrcommon cpp
fix a warning turned error about not returning from an unreachable point
switch to new history plugin
videoplayer fix setting caching at end of stream
dbg fix a bug with the debuggee notes being cleared on manual database save
add more rss debug
minor addition to previous revision
cleanup silence unused variable warning
fix bug where dirty bounds are out of the image bounds
dbg fix a crash with deleting all breakpoints
remove unused variable
add a test print to malloc test
better pattern printing
fix log router s starting popped version must be less than its starting version
fix if durability is on mods would have been journalled already
improve base_assert message so we quickly can find the file line
sema small stylistic change in initkindrequest evaluate don t use else after a return
upload queueing should now work a lot better
clients should seed the deterministic random number generator on the network thread
fix relaypriority calculation error
add batchindex in loading files
try fix merge with master
better sync cluster update diagnostics
address review comments
fix search from user in a group
event fix implementation of non sticky events
fix of previous commit
fix potentially incorrect memory access in cv transform
print warning if have full debug enabled
change sendcoins dialogue yes to send
swift reflection dump fix resolve call on branch
remove duplicated comments
fix layercolor bug for shader change
remove unnecessary code from qbtsession
expose share_weights to python to allow running test nets
use const ref args in probe_pt
httpheader parse fixes skip empty parameters use for m_protoline only needed part of string
disable function signature optimization at osize we might be more aggressive here in the future
update aggregatingtransform cpp replaced redundant call to chunk getnumrows with local var
show the linked action inside deleting linked authority error message
remove unnecessary include
core ocl add parameter to limit device max workgroup size used by opencv
add error string for cleanstack script violation preventing an unknown error if the cleanstack error condition is set
add add nfs to zeroconf browser
fix possible shutdown assertion with reindex shutdown credit eklitzke for reproducing
clarify rpcwallet flag url change
fix query for filtering movies in sets with a where clause thanks to montellese
fix min_marks_for_seek for the data skipping indexes
use pthread_attr_init instead of memset
fix erroneous break of supervision agency updates
don t run movetofrontbenchmark
fix column sorting restoring on startup
fix gpu perf tests videocapture reopen
fix warning ignoring return value
revert toolchain selection behavior
better error checking rs
bring window to front after restoring from systray
dispatcher comments self settouchpriority need to be fix
activeae fix compile warnings
hacked cpu interpreter to ignore branch on svc instruction as we are hleing this
recognize more binaryop with constant
increase minimum range in property editor a bit
fix latch sample build without contrib
update countingsortstring cpp
fix default concat dim
fix double whitespaces
remove nullptr access newbr here is always nullptr found with clang static analyzer
ae if we don t have a tick don t reduce any set delay
remove else clause remove the else clause cause bmfont cannot set its font size
remove unnecessary signal handlers
clear undo history on shader load
shift register buttons question can lcd shift registers keep up with faster mcus
service bcat check client connection before interacting with socket fixes a crash when bcat service is offline
comment out new options needs further investigation into performance impact
throw other errors except for eperm
cosmetic remove unused variables in linuxrenderergl cpp
some error reporting if skin xml contains invalid cursors
pvr get previous epg tag instead of previous of previous
add simple test for cbor byte
vcpkg portsdiff add missing const remove unused include
triggers an assert only ontouchbegan is nullptr for eventlistenertouchonebyone fixes comments for checkavailable
sourcekit don t handle operationkind all
comment out upgradable tests
no benefit from moving
sema remove some dead code from csdiag
now that locks are recursive readded access lock
renderer calculate present parameters before setting up fields we don t even need to hold lock here potentially
async cc pass task along from caller to callee
remove unneeded include
addons log checksums on mismatch
use change from devel branch
driver for the emit persistent pch job don t set an output filename
abort recovery when it looks like journal files are corrupt
merge branch master
remove repeated headers
fix files section size in dependency error dialog
vulkan fix regression with window resizing
sema don t desugar the type when diagnosing unexportable clang function types
rss code clean up
fix log routers were starting one version too late
subscore close search window if playback is stopped
fastrestore fix duplicate type in traceroleversionbatchprogress
merge remote tracking branch origin master into master next
edl look for edl files not only for smb but also for nfs and afp
merge remote tracking branch origin master into master next
stats only update process wide counters for frontends
fix ubsan issue
allow json rpc and eventclients to send real ascii unicode characters
gamelist supply default settings for qsettings config
addons group the texture invalidation under a begin commitmultipleexecute to speed up repository updates
make container node aware of size flags
attempt to fix search engine on windows
fix warning on vc
constraintsystem detect and diagnose assignment type mismatches related to existentials
fix merge bug
droid fix build this header may be needed elsewhere as well
fix grammar in comments
merge remote tracking branch origin master into master next
simulate real cameras by working around vtkusertransform to set projection matrix
remove unneeded register_plugin
vcpkg build print the missing package s arch along with the package name
clear sensorless homing in home_delta after first move to top
fix possible crash when deleting a torrent just after pausing it
add producer name to double confirm assert
check uniform set validity before freeing
remove debug output
fix loading localization file l not working regression
audioeffectfilter lowercase db property to db
update parsedtemplateformatstring cpp
bench replace mempool globabl with test_setup mempool this is a refactor since they are aliases for each other
extended optimizer rule to detect more or to in transformation opportunities
remove commented out code
ignore implicit conversion warnings
make find in finder result s text direction follows the search results one
add vector_vector_keypoint support to python wrappers
make update spinner popup use radio items
remove comment and update the tojsonhelper for indexrangenode
finish eliminator port
properly delete tmp files
catch exceptions when large allocations fail
show user asserts in verbose mode
minor unrefd local variable
account for minibatch per epoch
after converting sprite image type extras are regenerated
fix librtmp get bytes bug
progress bar minsize now checks percent_visible allows for really thin progress bars such as for pixel art styles
update mutationsinterpreter cpp
visualscript crashfix when returns are too few
add verification_cooldown enabled check
fix init segfault where initloadwallet calls atmp before genesis
make begin behave correctly when there is oom
buttontranslator check getfallbackwindow on getaction unify with joystick handling
remove unused ftry from push_lock
gsb compute minimized sources more generally
core ocl don t expose exceptions from opencl callback to avoid silent crashes of opencl worker threads
script create diag show extensions for currently selected language only
skip broken unit tests on windows
fix particles tangential acceleration randomness
add path cpu sparse dense dense to multiplyandweightedadd in matrix cpp
fix onplaybackresumed not getting fired anymore
was missing clearing computed uvs sorry realized late
prevent users from moving instanced children
cjsonvariantwriter fix possible null pointer dereferencing
fix merge conflict
drop deprecated hack
build sw disable fma dotproduct
replaced an if statement with an assertion in silgenexpr
fix float compilation
restrict cstdlib import to cygwin
videoplayer vdpau fix c p error
add blank line at end of file
eliminate an unnecessary use of astcontext getorcreatecanonicalgenericenvironment genericsignature creategenericenvironment is a better api for this
softbody add meshinstance material correctly
fix cross compilers
fix dynamic font crash if no available font size
use angle multisample extensions for uwp
reproduce mpm slp bug
remove unnecessary include
child window with menubar use regular windowpadding y so layout look consistent in child or in a regular window
minor formatting change in selecttilecommand
cost estimation boundaries description updated
why not search _toaddedlisteners
remove code that flips normal if facing away from test direction
ensure we clear art prior to setting up the list for re fetching scraped info on thread
merge remote tracking branch origin master into master next
fix skipping message with mediawebpage on keyup
add project message
updated from raw array to use vector
silgen update prolog code to emit unreachable if a func param is uninhabited
fix broken emoji
add a true argument
fix a wrong allocation for a tri_aql_node_t
rle dse avoid another whole traversal in ndebug builds in rle nfc
constraintsystem avoid recording duplicate fixes for contextual type mismatch
fix build command not printing out the missing dependencies on fail
fix regression running current scene unsaved will ask for main scene
merge remote tracking branch origin master into master next
fix use of progress dialog lines
egl render fix accidentially changed egl config
temporary revert to the old behavior
fix add missing filetimetolocalfiletime implementation in order to handle utc dates correctly
substituted the nan check with opencv s implementation
globalopt eliminate unnecessary coercion to silvalue and improve some small other nits
fix merge issue
baserenderer make viewmodestretch16x9nonlin fall in range
complete fix for windows compilability thanks volzhs for testing
fix undecorated name being uninitialized
fix build errors
removes unused cast in cctmxtiledmap cpp
gui possible fix for an irreproducible crash
fuzzer reset default database before reconnect
typo elif instead of else fix
reduce watermark console spew
change the newline character for csv export depending upon the platform
make rulers on by default
remove loop macro
fix version path
remove redundant lock
fix double space
fix bug in movingcelstate onmouseup restore the mask visibility when the position is not changed
sil simplify some code in sil verifier now that optional payloads are lowered
update ccparticlesystem cpp
merge remote tracking branch origin master into master next
vcpkg remove flushall
need to call xmlinitparser early its required if you re going to be using simple xml from multiple threads
don t recreate tray icon if it is disabled
generate gain and cover
zip allow libdvd to view zip root as directory to enable playback of zipped dvd files
only check if items are iso and expand disc image when in file view
sil optimizer remove now unused static function nfc
fix valgrind issues in persistence functions
add columns if modifying tables
nfc fixup resolvesilfunctiontype to not return type
update imgui cpp
ae fix mismatching allocation and deallocation of buff
add exception what to the thrown connection exception
change copyright header of pointer test
update zookeeper cpp
fix don t crash if we can t load libmad
fix compile bug
some help for top command
fix possible legacy path translation crash thanks vdrfan
add a separate string for empty channel admin log
merge remote tracking branch origin master into master next
cosmetic fix missing default switch statement
log when can t connect to arbiter minor
service hid npad fix implicit fallthrough errors
fix assert failure should be backup s contiguouslogend maxrestorableversion
change the resetrnn to true by default in extendedeval
stepper indirection fix tmc driver
merge remote tracking branch origin master into master next
fix null pointer access
pvr ignore article folders when sorting timers
pvr remove unnecessary condition it is checked inside seticonpath
fixes resources name not showing in the inspector
addonbrowser remove date sorting doesn t work
fix linux build
dbg fix incorrect efree in command
set default locale
tests add fuzzing harness for count_seconds
remove some prints
jsonrpc fix mimetype being empty in files getdirectory
gardening use correct copyright years
add a fix
fix editor help scroll to result
globalopt add assertion for resilient classes
merge remote tracking branch origin master into master next
fix getartistsnav query
fix double free for drag preview control in viewport
remove a trace event which was causing valgrind errors
dbg correctly remove librarian breakpoints on exit remove hwbp on detach
fix whitespace issue
remove an unused var
dvdplayer make ffmpeg vda default
fix mac build
fix system tables output
don t use utf strlen if the text is not utf
fix format overflow warning sprintf needs an additional bit for the nul terminator
remove rendergui var from posix main was never used anyway
nfc remove unnecessary radar reference
fix rpc return type
sanity check test
softmaxlosslayer fix canonicalize input axis
plot use default location from preferences dialog when saving picture
conformance checker eliminate redundant recording of missing witnesses
evaluator test pull in std isnan
increase timeout for dropping collections
don t do dbprofile if fsync locked
sqlitetablemodel return back cheap blob test for data displayrole
on fork should be true if not found
animation bezier editor fix double inserts on undo
paplayer add missing calls to iplayercallback s onplaybackresumed and onplaybackpaused
gui restore rpc console to non wallet tray icon menu
improved keyboard shortcuts
include what you use for abort
better help method for internal command
update threadstatus cpp
fix gethost for config servers
remotemirrors remove now unused code
improve polls layout
update split lines uses the edgemode if enabled
am cleanup istorageaccessor read
update bagofwords cpp
pvr cguiwindowpvrguide cleanup remove unused define
merge remote tracking branch origin master into master next
fix only check video playlist status to determine fullscreen playback
fix caret not drawing on focus regain
amlutils additional checks on sysfs for correct functionality on amlogic platforms signed off by sam nazarko email samnazarko co uk
gui add check for valid memory address of thread entry
pvr timer settings dialog add weekday for dates
info only extract thumbnails and video info from real video stream
small fix in comment
update expressionanalyzer cpp
always respect getrequiredfee for wallet txs
use return false instead assert in cwallet signtransaction
don t lookupoverrides of special names
don t spam logs if there are no dbus
import wrapped dropout subgraphs from tensorflow
fix formatting of comment nfc
fix compile of charsetconverter tests
osx fix include in coreaudio cpp for case sensitive filesystems
yuzu main fix compiler warning
fix c p error in crendercapturegl
include oplog entry in fassert about missing version number to help diagnose test failure
add missing max number of samples msaa check
irgen remove obsolete todo
ccx trace accelerometer only when tracing rendering
ffmpegimage free avpacket side data
either comment made no sense or i am confused
add a new line
don t clear members in destructor it s not necessary
raise main window when starting another instance w o params
clean up misc tovalidfilesystemname
fastrestore fix testing on addprefix
update threadfuzzer cpp
capitalize closure php is case insensitive but this leads to better debugging messages if the case matches the declaration
irgen add an assertion in enum lowering
fix typo in visualshadernodesample3d generate_code
fix total fps evaluation in the gpu hog sample
pass mouse events to spinbox from its lineedit
remove fdaemon flag checking on return from main which is useless and looks really strange
use ref release to release ref instance
resourceformatloaderimage get_resource_type now uses file path
if a collection cannot be dropped during restore truncate it
project history was not being saved after a recent fix this should fix it
ast declcontext dumpcontext prints to llvm errs by default
activate app explicitly on os x on debug mode
obj importer fix relative path materials
update storagemergetree cpp
fix particlesystem _paused varible isn t initialized particle will show or hide after being created randomly
update context cpp
fix mirror property of kinect
make udp server scenario less fussy
videoplayer fix subtitle on off setting
clear trx_in_progress_size when closing a connection
prevent autotile properties out of bounds
fix dirty read of objectid counter when threads are involved
add uppercase and lowercase to rightclickmenu in scripteditor
mysql properly toggle mysql transactions automode
ios fix crash case check for null m_fmt_desc before checks that might release it
track upstream llvm changes
use checks check_exit
cguicontrollerwindow fix focusing of dialogok if no peripheral addon is available
better error msg
use errorcodes unknown_user
set cpu frequency before running setup
fix line thickness for collisionpolygon2d
attempt to get rid of compiler warning
merge remote tracking branch origin master into master next
fix cpu only build issue
disable potentially unsafe socket option
remove an un needed apostrophe from a user visible string
fix another dangling reference problem
fix a cppcheck warning
gui fix window title update
update users cpp
call socket postfork in case we need to finish accept
decrease verbosity of disconnecting clients right now they go to error log
merge branch master of github com mongodb mongo
xbmc save de init order of services if manager is destructed not in designed manner on closing application forcibly
sema fix quadratic checkfunctionerrorhandling in wmo mode
change gpu acceleration setting before the display scale
don t access nullptr
csdiag remove obsolete code block in visitcoerceexpr
fix uninitialised bytes in call to write
raft timings to mean rather than max
ffmpeg fix iocontext memory leak checked with valgrind
chd changed position of a codeblock to not count frames twice thx fritsch
force progress update on torrent pausing
add license return attribute
remove unreferenced variable breaking build
cguidialogvisualisationpresetlist align header label and list control ids with settings dialogs
sema add a couple of comments
android fix crash on kodi exit powersyscall no more available but called
omxcore skip out of flush on error once an error has occurred flushing tends to time out and cause delays
code clean up
dvdplayer reduce likelyhood of jumpy pts causing duplicate subtitles
rbp fix omx use sendcommand from component in tunnel class
loader lower file name logging level
pvr fix bad erase in cpvrchannelgroups deletegroup
fix error when duplicating node from editor fix error when duplicating node from editor
update pool cpp
fix wrong message
add clarifying comment to foreachinputnotproducingsupplementaryoutput
serialization register imported_module_spis in the block info block
common fix conversion warnings
revert aml fix draining message queue this breaks ff rw seek
elim ambiguous call warning error
fix undefined behavior
merge remote tracking branch origin master into master next
fix typo in string
fix redraw in sd card menus
style and nits for fee checking in broadcasttransaction
update storagereplicatedmergetree cpp
fix compiler warning
net misc header cleanups
avoid compiler warnings
turn on sni in openssl
remove debug logging
clear the input line after activating autocomplete
update regexprowinputformat cpp
don t disable the ssl feature with the restserver feature but the ssl server feature
update ccfileutils linux cpp
add unit test for prelu layer
elim warning on windows
reflection fix accidental edit
fix do not process nfourl functions twice thanks cptspiff
minor fix on printf datatype mismatch in sgd
cdatabase fix return value during createdatabase
change assert to warning for now if recursively locked
langinfo remove unused code
make journal dir earlier if dne
fix minidump logging output
instantiate caffe_cpu_sign for float and double
cleaned up example slightly
encoder signed vs unsigned warning
correct order of colors in the bottom line
sourcekit put sourcedocinfo astprinters into anonymous namespaces nfc
fix sizes of columns doesn t match error with simpleaggregatefunction
adjust decimal precision increase the number of significant digits when converting from double to string
ast fast path for requiresnewvtableentry when called on non class members
try to use another namespace for simd version
ignore whitespaces in dice sending
fix windows compile use mongo polygon instead of just polygon to avoid collision with windows gdi function polygon
upnp art fanart thumb images over upnp to use previously cached art
parse use scopekind enumbody not scopekind classbody for enum body
rss handle more types of rss feeds
remove unused check
fix crash when checking empty string for valid hex number
python fix to allow pydev debugger to work
fix bug of action manger ex
merge remote tracking branch origin master into master next
set default file name for exported projects
fix verison number conflict
bench fix negative values and zero for evals flag
wallet optimize rescan to skip blocks prior to birthday
apply clang format locally
cssolver update expression after constraint generation while solving for code completion
runtime fix trivial linux build failure
remove each time new version notification
revert iuo convert an explicit check for iuo to check for any optional
remove stray semicolon
remove coordinator printf in simulatedcluster
make the editor dimming even more smarter
just initialization needed by compiler
backup expire will fail earlier if force option is needed but not specified
slight log cleaning
merge branch cebolan fix undo history
demuxer set correct startpts when seeking chapters of discs
fix possible memory leak in varfree
stitching perf increase test check tolerance
fix gesture events being blocked
extend the error reporting during file mapping operations so it includes the address where mapping was attempted as well
log a little more
fix top bar layout when emoji section is visible
avoid assert in lapac cpp if findhomography fails in bestof2nearestmatcher match
use rslog for log msg minor
fix warning in opengldriver in release builds
fix semicolon parsing in visual shader expression
vda getpicture return bool
autorelease custom extended class fix memory leak
fix sorting in finished list
remove duplicated include
feebumper discard change outputs below discard rate
remove initialisms co authored by a j beamon ajbeamon users noreply github com
dvdvideocodecffmpeg log filter graph for debugging
createtransaction don t return success with too many ancestor txn
citra qt small cleanup
just use a char
improve media viewer window decorations
add missing symbols for solaris build
improve message of vcpkg integrate install
windows prevent huge prints from crashing the engine
fix compile warning
bugfix make ccoinsviewmempool support pruned entries in underlying cache
fix pass a cfileitemptr to the announcemanager so we end up with the right content in the actual announce
document unchecked invariant signed off by edward z yang ezyang fb com
dvdplayer set cmediasettings get getcurrentvideosettings m_audiostream on open set
minor code format fix in rpc related files
restore original cccopying as the parent class of ccobject
use the proper fee function for json structures fixes memleak
changed to creating the slice diretories if it doesn t exist
add const to rocksdbkeyvaluestore methods to support new ikeyvaluestore interface
fix for lazy database
retrieve windows error string
fix querypipeline unitepipelines
memory fix typo in rasterizerflushvirtualregion
gardening fix recently introduced typo an swift a swift
fix vis control rendering was off due to setting invalid viewports
fix conversion warning
fix last commit still had some debug code in there sorry
only request_message for our head greater than msg head when we suspect they are on a fork
make clang tidy happy
re enabling matrixvectormax test
fix a bug that demomodernart crash on bada
update mask appendix eventtap
gui define max_digits_btc for magic number in bitcoinunits format
fix swiftpm targets without swift swiftpm will pass a runtime compatibility version none in this case
sqlitetablemodel speed up on large table in certain circumstances
update unit test
fix overly restrictive assertion for dump scope maps case
makes the group dialog resizable
remove trailing whitespaces from formatted queries in some cases
additional unit tests for eosio account name prefix and account name length
gardening remove unused variables
readline vs linenoise vs dummyshell next try
gardening fix spacing
remove double quotes when saving wallet password to file
disable cpu temp test on windows
add swiftremotemirror implementation for addreflectioninfo
remove unused variables
add unit test
canvasitemeditor changed frame selection shortcut
physicsserversw body_is_shape_set_as_trigger missing return statement
remove long since disused define
fix display sendfilesbox via historyattach button
wallet fix accidental use of the comma operator
add an assert in case of write into a file after close
fix test_model detectionoutput
generate document in rest handler now works with externals
irgen streamline integer literal emission
test fix message for ecc_initsanitycheck test openssl is long gone
remove unneeded cast in rpcmisc cpp
videodb fix getting streamdetails in getfileinfo
fix use the proper type for cstdstring npos checking
macos fix crash on failed fork
fix video player and web view in js template
gui and finally done correctly
fix a bug in neighbors matchvertex it aborted too early some times
mbl z adjustment on toolchange
hide fg bg color selector dialogs when we are not viewing a document
one liner fix for crash line was remove accidentally
tests unittests mathtests batchnormalizationenginetests cpp bump a tolerance
cec don t update the device type in settings with the value that was returned from libcec
fix bugs in this test
idle during long arcs idle during long arcs to prevent from watchdog resets during high segmented fast arcs
fix function names
improved arebackgroundmovesneeded method
updated the unknown name
make cppop autograd execution work temporary signed off by edward z yang ezyang fb com
fix up styler s grudge
adjust some hal formatting
remove a workaround for zlib ng memorysanitizer problems
boop the ci
skip vmreganchor for argetcontextclass
swapped tid and game name to make it easier to parse
elaborate on the purpose of response files
fix tv vendor override typo
copy support chunked reads so that cfile internal buffer is eliminated this eliminates memory copies cpu overhead by multiple calls
all fix restore remote keyboard feature thx memphiz
remove bits stdc h
support for quorum
make sure phantom exit really exits patch by alessandro portale
fix shard version caching
download queueing fix
fix for text field
add group privileges for files created conflicts lib basics json cpp
aesinkaudiotrack document truehd static fields
skip retrievemusicinfo for smart playlists like music nodes we already have the info
basedaemon change terminate signal to sigterm gdb catches sigint and by default don t pass it to daemon
fix small visual bug in filesystem dock
corected condition for piddt
smtp fix finding the local address
memory leak fix for clippingnode
rbp fix channel mapping for hwdecode path thx popcornmix for finding it
fix compile error
fix crash in pluginscript complete_code
addressarbiter fix bug with break statements missing from case statements
videodb bump version
cudnn rnn fix typo
win device notifications third time s the charm
forgot to call one script with python
fix windows build maybe
fix problems in python doc generation
diagnostics add a tailored diagnostic for attempt to use optional as a bool in argument position
merge remote tracking branch origin master into master next
peripherals fix locking wrong critical section
do not demote dense struct for with bitmasks
fix bitmap calculating incorrect true bit count
get cancelable for gcode cpp
use lsn file outside of debug mode
use msf not maininputfilenamefordebuginfo
remove useless condition
better message when root missing minor
ml fix caching of internal state when changing the impl in knearest
fix nextid in clone method for execution plans
fix segfault in debug mode caused by previous commit
net update getnetworkname with all enum network cases
moved scalar variable value to int main to work around a gcc wshadow warning in operations hpp
try to fix shutdown issue
print logs to stderr
osd print script name instead of meaningless script id
support positioning dialogs via left and top
when hls timestamp jump reset it
fix concurrency issue cleaning up pooled scopes
fix custom loader saver broken
fix gridmap palette remaining invisible
initialize _name on posix
lower thread start logging level
upstream update clang tooling replacements insert clang tooling replacements add this is just a rename
remove commented code from combobox cpp
wallet fix clang build in mac signed off by anthony fieroni bvbfan abv bg
update comment based on cr feedback
define the default remapping in the right scope
bind missing constant in visualscriptpropertyget
fix an out of bounds vector access
fix a bug with double move which corrupt original part
add back the mutex holder that was remove accidentally
fix joystick axis handling for input is_action_pressed
rare segfault fix
penalize backwards sorting
rescan files and folders deleting them
always use connectremoteudp when using udpip connection to remote hil this sould also work in the localhost case
fix crash when change project settings
fix minor warning
cimpile idset minor
fixes to clippedcamera this work has been kindly sponsored by imvu
update storagedistributed cpp
upnp disable notifications while cleaning dbs
fix build error
astc call std vector reserve on texelweightvalues to avoid reallocating
use scoped_ptr not shared_ptr in fuzzfile
fix an issue on hc timing in ut
default codec if part is not stored on disk
improved tracker deletion code slightly
missing check_nothrow in unit udt
diagnostics tailored diagnostic for missing arguments in contextual conversion
implement range check in uriutils resolvepath
error handling in rest handling and random generator in constituent
fix cuda vprintf namespace
add assert for valid w values
non clickable mention in changelog
fix linux autoupdate for dev channel
addressed jeff s comment
lua add negation operation unary into point metatable
richtextlabel clear the scroll offset
merge branch checkpoint
merge branch master of git github com bitcoin bitcoin
fix read from invalidated iterator in callbackhandler cpp
add test for mat_ release
osx mavericks on mavericks don t use pbo rendering with intel gpus as it might result in distorted rendering ripple
to set csd version for serializing
silgen add some comments to callee s constructors that explain which constructor corresponds to which factory methods nfc
cosmetics add missing space
fix unread badges for new dialogs
aesinkaudiotrack remove headphone check let audiotrack handle it internally
fix longpress do not repeat long presses
osx ios fix video playback reduce timeout
coverity initialize videoinfotag rating and uniqueid ids
remove unused variable
donot release stream when busy
replacing not implemented query
rasterizer memfill set the correct stencil write mask when clearing the stencil buffer
bug fix in bs definition of times macro
prioritize first and last pieces when sequential download is enabled
fix potential int64 overflow in int array conversion
avoid closing brushpopup when we click the menu popup option
pictures fix video in slideshow
add override add onchecked function to timeline command
temp make setnojournal use journal to try to fix buildbot or at least isolate the problem
fix compilation with mingw
colorpicker margin bug fix
gardening fix recently introduced typo irredutable irrefutable
fix warnings building swift frontend on windows using msvc
fix dependencies of init database feature serverid feature
addition to prev revision
switch ngraph op v1 multiply to v0
fix tween transform2d typo
rs skew check
pvr fix crash in pvr search window context menu
fix handle removable drives for media window title
optimize getoldestkeypooltime return as soon as we have both oldest keys
simplify relooper default indent
add roles to health status
fix shutdown issue
fixup order check for m_audiotrack_empty_sec_tweaks
incremented version number
clang importer compute requirement signature before setting environment match what the type checker is doing
fix clear m_profiles later as setcurrentprofileid uses it to set the profile replace another size by empty
update jsoneachrowrowinputstream cpp
changed return of locks
add a final lift if configured in manual bed leveling
remove unnecessary variable
getwritersnav tried to use the studio link table instead of the writer link table
remove duplicate rpcsslciphers help message from init
sema designated initializer overrides are fully type checked
fix compiler warning
fix memory leak in load_document
drop extra setting of currentmenu
fix if condition
check that logging framework is initialized early to avoid segfault
remove trailing semi colon
show blocked restricted users to all admins
unused variable res
update storagemergetree cpp
add missing guard
fix lambda function for curl code errors signed off by stefan weil sw weilnetz de
remove unused code and use nullptr for pointer
mmalrenderer use half rate deinterlace when playing video in background
fix if the known committed version is equal to the recovery version we do not need to copy any data
report warning on fastrestoreapplierclearrangemutationsstart if delaytime is too large
adjust error message
resolve ambiguity in mutex declaration
fix sftp stat didn t set directory file type causing eg libdvd to fail on isos
fix classify chars misc utility bug caused by cropping outside of bounds of image
bpf lower the default memory settings
physics area add monitorable check
enable eye dropper in color button widget when we drag drop the mouse button
update spdlog cpp
better index debugging
edittabledlg don t react on textchanged on every text change in the type combobox a renamecolumn was triggered
remove debugging cout statement
linuxrenderer remove obsolete includes
fix gdscripteditortranslationparserplugin instance leak
add comment for pending changes
remove unneeded sort from spriteframeseditor plugin
update xbmc guilib guipanelcontainer cpp
hid reset acceleroeter and gyroscope index in init
log database version to logfile
vi change transactionid cancelbuffer to log_critical
fix crash if passing invalid image ref to texturelayered
add missing initialization for m_forceunicodefont
fix a test assuming cpu mode without explicitly setting it
utils cleanup includes
label change default font size
test fix headers
fix a bug in aql replace
tests add amount compression decompression fuzzing to existing fuzzing harness test compression round trip
json rpc return empty for fanart if it doesn t exist
fix check for arguments at xhr send
fix missing argument to printf in fdbserver
don t waste time jit compiling isolated functions this is already done ahead of time when building the executable
add mpaa rating support for json rpc thanks giftie
fix crash when run testtimelineframeevent in cpp test
pgo new er ihf heuristic
reduce number of stat syscalls for mergetree data parts
update storagefile cpp
fix issue with simpleaudioengine geteffectsvolume for linux simpleaudioengine geteffectsvolume was returning the background music volume when run on linux
clanginfo fix loading of m_languagecodegeneral on linux osx
disable error throwing in convexitydefects when hull is a line or point
remove unused code
net remove unused set
add description for replicapartialshutdown in system events signed off by slach bloodjazman gmail com
wip static lib
forgot to remove debugging code
explicitly inline trivial function in cumulativereporterbase impl
don t create a new db to test auth
fix hls muxer segmentation fault
extendedprogressbar fix possible division by zero
revert fix bad color to html conversion alpha channel was add before rgb
fix bug in ippmorphop function when looping over elements of the morphology kernel
remove unused argument names
ast fix formatting nit
allow existing directories during init
fix active in curop
fix error while json parse with empty string
sema re implement getinheritedforcyclecheck for protocols
add newline to end of file
ae pulse fix improper usage of new caewavloader
videoplayer fix blu ray playback in menu mode after changes to streamid
fix webgl build
fix vc build
input adds support for window_fullscreen_livetv handling to getactivewindowid for gamepad and joystick devices
update context cpp
threads remove a redundant function use the next_thread_id variable directly
fix if selecting a subtitle strea switch on subtitles as well
rs gcc compile
add a comment
osx fix compile warning
make bsondump check for empty path
refactory the code
add the default arguments to the binding for resourceimportmetadata add_source to match the c version
timers return an error when calling settimer with negative timeouts
disable forward button correctly when the file selector is opened
activeae reset fading flag when drained paplayer waits on this
add new macro for force function to ram move uart_interrupt_handler in ram
make sure we process the hidden focusedlayout when an item is unfocused as this ensures we mark as dirty
dyncom actually set the q flag for smlabb smlabt smlatb smlatt easy skyeye todo fix
pvr the channel used for updating channel spinner wasn t correct
fix lines clockwiseness to be consistent with other primitives
remove test output
fix temp var issue
update parsedtemplateformatstring cpp
remove redundant return
upnp fix additional slash at the end of container uris
updating usage of sanitizercoverage
properly save the new save safe setting avoid crash
same method when redrawing as when theme changed
fix main menu item location on macos
change path of user extensions to extensions instead of data extensions
audiolibrary check boolean option values
changed remove an unused variable
update src formats jsoneachrowutils cpp co authored by tavplubix tavplubix gmail com
bridge fix a terminating error message when there was no settings file
add try catch around boost filesystem exists to handle inaccessible path signed off by matt kangas matt kangas mongodb com
videoio realsense add cvtcolor as the pixel format is only a hint
make button and co take internal margins into account when clipping text
mmalcodec report when support is compiled in
fix adaptivethreshold mean
use appbase with fifo priority queue priority queue in net_plugin no longer needed
core forbid handling of the case when src dst in cv repeat
merge branch qoega random table function small of github com qoega clickhouse into qoega random table function small
advancedsettings regexp fix ignore itunes extras directory from movie tvshow music scans yea my regex skills suck
clea hid_observers_ before hid_manager_ stop
dbg fix a crash in plugins on cb_breakpoint
commented out unused debug variable to silence static analyzer warning value never read
merge remote tracking branch origin master into master next
merge remote tracking branch origin master into master next
fix sercurity issues in driver
yuzu configuration show assembly shaders check box
update interpreterdropquery cpp
fix linux build
more strict aliasing
rpc check that left section is not multiline
fix failing tests
indent test output for easier reading
paplayer prevent possible division by zero in mp3codec
fix powereditor compiling error
fix bitwise and intended not logical
dialogcontentsettings fix refresh logic on scraper change pointer comparison is always false
add a little tiny comment to accessenforcementselection
remove unused variable
set default color to black
fix the build on clang signed off by edward z yang ezyang fb com
update libkrbn for scoped_dispatcher_manager changes
qt tweak status bar styling
the code to convert der bytes to ecdsa_sig has been changed
make key_size a compile time constant
fix argument value to correct type
invalidate cached art when refreshing video items
cleanup silence format security warning
testgradientbasedsolver drop doubled seed inititialization
fix unused variable warning in release mode
stdlib fix an unused function warning on linux
fix the way to get tagname
fix typo in read lock accumulation
pvr guide info dialog fix record add timer button visibility
add default value for _batchnode in ccsprite
dsp hle audio output
merge branch srs master
fix text reader binray version setfontname
fix a compile error in case of building without highgui
cli add multiwallet capability to getnewaddress and generate
videoplayer drop error mesg failed to get stream it often fires false positive
cosmetic remove unused propertys from airtunes these are maybe able on boxee but not on xbmc
remove spurious test
fix inadvertent rename to original test function
textfieldttfdefaulttest fix can t take input from the keyboard
in setcliprect remove top and left positive only limits
rpi cec and now correct use cec_rpi_virtual_com from cectypes h which also uses the correct value instead of cec
windowing gbm gl fix log leftover from gles
toggle use_gdbstub in citra glfw
fix cppcheck complaint
apt_u add event creation to initialize method
remove non special comments
only send reorged txn to mempool after checkpoint
numericdialog no need to have an ip address to pop the dialog up
fix add missing begin and end member functions to alt_string
fix check for killable processes
comment unused function on macos and ios the unused opencvbigtohost32 is a warning for buildbot
pvr timer settings cleanup remove unused variable
gif skip frames with an illegal dimension rather than abort
silence complier warnings
trivial bip bip in help text and comment
fix memory leak in rss parser
add code coverage macro to datadistributiontracker safeaccessor operator
updated linked list with in pre post order traversals in c
don t serialize dynamiclookup dependencies in modules these are effectively uses from the perspective of a client module
fallback to the name in the attribute when found duplicated candidates
fixes wrong calculation of gridcontainer s children size
no sleep when we have job
fix undefined behaviour of doubledelta
bugfix off by one in priority calculation
silence to debug messages
add comment to clarify the nullptr check
header comment add
fix crash when the disassembler pause button is pressed while no game is running
remove mini style from opacity slider in cel properties dialog
small treeview optimization
bogus matching entries warnings
fixes crash when loading streamtexture from file
executor needs to check its inital map of code
fix compiler warning
quell signed unsigned comparison warning
vcpkg create minimal package structure even on failure
sema simplify typeresolver resolvesilboxtype
change round return type to float
fix the fix of the arangodump bug in cluster
merge remote tracking branch origin master into master next
standardize usage of templates in matchertests
videoio images fix buffer overflow problem
move connection info
fix a logic error in actionscaletest
update diagnosticverifier cpp
handle sigquit too
fix build without ssl disable mysql protocol support
fix memory leak in shoutcastfile cpp
videodb add missing break in switch case
fix hassatellitereplication was set incorrectly
simplest fix to a dependency bug better commenting out
remove a debug print
using new dbdirectclient
gpu test output files moved from cwd to temp folder
deglobalize system am
add rollupstep and cubestep
fix totals without having
ensure parameter type
fix small typo
fix columns lines not being pixel aligned
win more informative message for old chd driver
test add tests for self usage in arith_uint256
ignore qb when extracting file extensions
media library only apply the advanced filter if there s something to filter by
fix a shutdown race
leave clustercomm performrequests immediately at shutdown
fix mruby action
clarify the error message when unable to bind to port
test running with only one region
csdiag simplify the ispropertywrapperimplicitinit method a bit
moved the variables in the isr thats the most normal use case of static
dbg fix typo
improve the coding style
improved code coverage
use switch case
remove unused code from getblockchaininfo
avoid crashes at shutdown due to printf in global destructors
make it more clear what the new button does
update flatc cpp
make sure to wait for core chunk move as well as delta
task cannot be scheduled and deactivated in the same time
revert disabled flaky test
fix effecttest crash
merge branch display version in help of https github com mhanne bitcoin
just revert return insertion if inappropriate because the bodystmts are revered to their original contents they are no longer implicit
add tags field to ismaster
fix wrong judgments of conditional compilation
codecs dxva changed deinterlacing method to dxva instead of auto auto confuses users
get rid of compile warning
bullet fix inconsistent layout behaviour when clipped
change border type for medianblur to border_isolated
fix test case failure
change cleanup unneeded casts
check for interrupt requests in tls config refresher
videoplayer fix displayed a v error on codec screen
fix closing float player over intro
guiepggridcontainer validate scroll offsets as well as they are used to calculate rendered blocks
lightbaker include stdlib h to fix ci builds
merge remote tracking branch origin master into master next
fix colorizing disparity
fix crash of performancetest
dbg allow stirng formatting in labelset
savedatacheck fix a typo
fix missing permission bitmasks on freebsd
wrap call to flush_presort
fft document usage and use std swap
use core extensions when available to set max anisotropic filtering level
stop excluding artist with name that matches the current localised string for various artists
do not process incoming transactions in read only mode
make sure we specifically set allowhiddenfocus to false in the python implementation of setvisiblecondition
report correct block num
internals renamed focuspreviouswindow to focusfrontmostactivewindow
update parsertablepropertiesquery cpp
add removal for backupworkercache
don t use qt wheel scrolling
remove cstdio debug
describe maxmempool and mempoolminfee in the getmempoolinfo rpc help
update datatypedomainsimpleaggregatefunction cpp
fix crash in cshoutcastfile close clear buffer pointer after delete
gif check if there s a frame before accessing it this prevents a possible access violation
sil getabstractionpattern var never returns opaque pattern nfc
gtests disable event grouplimitedgroupscope till somebody fixes it
fix setdebug functiont to apply correct value
qt change default size of intro frame
searchtab can now save sorting column changes
avoid visual studio compile warnings
mediacodec add hw accel hevc playback
fix hanging when switching focus between different editable comboboxes
minor initialize callback
merge remote tracking branch origin master into master next
fix compile error
cosmetic more compact and faster initialization
fix missing brackets around else case in cnetworkinterfacelinux getaccesspoints
attach to current thread in threadpool
rpc docs gettxout clarify bestblock and unspent counts
refine code for log
serialization fix a prettystacktraceentry typo no functionality change
merge remote tracking branch origin master into master next
merging dcgan changes that needed to be refactored from older primspec approach
ae wav fix remap not remapping the resampled frame count
fix compile issue when watchdog was disabled
fix clang warnings
upnp do a full loaditem from thumb loaders instead of filllibraryart
downgraded log message about unknown suffices
if the user passes an empty output matrix initialize it to identity
dbg allow setting temporary labels from the labelset command
fix compilation error for old libpng versions
skip background class in gathering
activeae fix incorrect return values of getspace
remove streamdiagconsumer to prevent crash
merge remote tracking branch origin master into master next
mark down an alias
videoplayer fix double call of preinit and missing render formats for sw decoder
merge branch master of github com mongodb mongo
sw remove inexistent include dirs
ast fix accessor end locations they shouldn t involve the parameter list because they aren t present
fix shutdown of system_logs
make hot region bigger for toolstrip popup
resolve merge conflicts
commented off print due to the report from the mailing list
make msgassert print stack trace maybe get rid of not sure
temp runaway proper cast to prevent any overflow
constraintsystem use the new getconstraintlocator overload
fix currentrate calculation for local ratekeeper
add the return value
fix addons cast lastinsertid to int
fix tv shows quick path missing
ensure file export table s to json is enabled disabled along with its siblings
slight improvement of the notification code
rle dse refactor a variable declaration in dse nfc
remove warning msg
fix arguments for tprintf the format string expects two int arguments signed off by stefan weil sw weilnetz de
cosmetic tab to spaces
duplicate groups and signals
mipgen now creates the out dir if needed
add access control allow origin to http
remove unused variable
fs implement mountsdcard
ast don t emit anyobject aka anyobject for primitive anyobject
segfault just introduced recently fix
do not count filtered values twice
cguimediawindow fix compiler warning warning was member m_result was not initialized in this constructor
fix linux compilation
wayland actually read from event interrupt pipe to fix cpu hogging
dvdplayer open default streams after demuxer changed streams
merge remote tracking branch origin master into master next
cosmetics fix c p ed code comment in cvideoinfotag tosortable
increased the proxy commit memory limit
fix error message about granularity
remove assertion all together and use getaddressofstackinit instead of dyn_cast to storeinst
aesinkpulse don t throw packages away if we are not yet ready
use qfontinfo so windows chooses a correct font
addressed final code review comments
remove unused phi define in math funcs
fix gcc warning so buildbot is happy
fail early with compound shard keys
remove direct tiff writing to in memory buffers if libtiff is used
fix download icon in peer list popup menu thanks heavyjoost
remove thread limiting
fix minor unit test memory leaks
gpudummy cpp should not compile if cpuonly
update comment todo for sequence header and fast startup caching last gop
fix display scale bug in visual shaders
bugfix unbreak build by defaulting to true
don t spam logs with missing keys
remove excess namespacing
make shard lookups for replica sets work correctly
enable googlenet alexnet vulkan benchmark fix build on msvc
add tests for hexstr corner cases signed off by kosta zertsekel zertsekel gmail com
fix doxygen comment the transaction is returned in txout
add a perftest
init arg1 as false for safety
armemu fix sxtah
fix ubuntu warning
fix amc hevc let cbitstreamconverter decide fixes hevc on android tv
getdrivetype require trailing backslash
delete unnecessary include
add log statement for queue write failure
wrap some macros
update utils cpp
dbg remove redundant null terminator assignment
optimize empty string comparision
get rid of query settings after initialization
flush stderr after each line of log
fix build errors on linux system
fix broxopticalflow sanity test increase epsilon value interpolateframes and createopticalflowneedlemap
add vertex unsignedint size rgba
fix memory leak
nginx never keep the metadata
fix nox compilation
remove l characters
intsimdmatrixsse remove unused include statement and simplify code signed off by stefan weil sw weilnetz de
biggest of n in cpp
switch g_windowmanager process to capplication framemove so that window close anims work as expected
thread cleanup minor
drop obsolete test for leading n
remove some annoying log spam
fix axis indices for coreyz
skip indentation of empty lines when indenting a selection
improve ui text and ui textfield creation performance
use nullptr instead of null
horisontal mwhell support
slave override for listdatabases
tests stub out more of the stdlib key paths need metadata stubbed out for the unit tests
merge remote tracking branch origin master into master next
assert on error writing file during recovery
update vulkantexture constructor for textureusage
addons stop service add ons on profile logout rather than waiting until the next profile logs in
potential fix for memleak
silgen simplify a check in silgenbuilder createstore
one more fix for pvs studio
sema update indentation remove unused index param use getinterfacetype instead of gettype
codecompletion zero check interval should always check dependencies
improved vload opt
fix grid view button icon not showing on filesystem dock
fix a bug in undoable copy_previous_frame using layerimage add_cel directly instead of undoable add_cel
pvrtimer remove unused variable
fix builtin playmedia playoffset param
fix incorrect iterator comparison in pvrchannelgroups cpp
use str equals rather than strcmp
linuxrenderer proper cleanup
update csvrowinputformat cpp
wrong method name
update subsetsum cpp
enable ocl version of minmaxloc with mask
switch to g_nondeterministic_random to make simulation deterministic
ceglimage allow logging if logvideo is enabled
add ricardoquesada info
activeaestream fix compiler warnings
bugfix sometimes app directly at the startup sometimes app directly at the startup on android platform
fix virtual keyboard was not centered
qt ensure socket is set to null in paymentserver ipcsendcommandline
rotate foreground color hue by default with mouse wheel above colorselector
remove invalid semicolon from aesinkalsa cpp
catch ctrl c and exit cleanly
fix a merge
default walletdir is wallets if it exists
swift ide test enable subclass existentials since i don t know how to pass langopts flags in
stacking skip folder stacking entirely in case no patterns available
fix style check declare error code illegal_type_of_argument
nfc downgrade compilerinvocation to frontendoptions
cosmetic fix typo being placed in log
remove redundant camelcase_to_underscore call in editorinspector calling string capitalize already calls string camelcase_to_underscore under the hood
fix issystem for timer tasks
fix build warning in ustring cpp on windows msvc platform
orb compute default norm by wta_k
dvdplayer put line break into video codec info because it gets too long
fix process free regression for libprocps
addons addons for linux don t necessarily mean they are compatible with freebsd
rbp omxplayer no need for two debug messages per component init
post to application thread no longer needed
bugfix blobdetector was putting incorrect diameter value in keypoints
use device orientation for acceleration values
update fdbclient backupagentbase actor cpp co authored by satherton stevea apple com
include pdfile h to ensure diskloc obj is defined
fix double counting of range reads in transactionmetrics
fix crash when destroying audiostreamplaybackoggvorbis
tests fix cxbmctestutils referencefilepath
patch command order in delta calibration menu
return correct language type
fix dpad left right not showing correctly in project settings
remove some useless log messages
remove duplicate includes
merge branch master of github com mongodb mongo
fix crash for large bw tif images
update datatypenullable cpp style fix
add probability_that_sequence_is_increasing to python api
corrected spelling of the word available
update registerdictionaries cpp
directxgraphics checking if unsigned variable foo is less than zero
fix imageview render error
hide only pluginscript in dialog box
fix asset library video play overlay not being centered on the thumbnail
header comment add
fix clang build
remove uneeded header
eliminate false positives the routine was incorrectly returning too many garbage rects
cs remove an unused variable
fix a typo
fix unitialized memory
merge remote tracking branch origin master into master next
remove clever use of std ignore
fix help view not scrolling to the top
fix a window build error
fix crash using newbrush when home tab is selected
use larger buffer and specify size correctly
dvdplayer only set audio and subtitle indax on user action
fix assetsmanagerex small issues
a few minor changes in the blockrandomizer output
modify the sequencecompleted method in ccbanimationmanager so it can set the next sequence in callback
disable debugging output
fix atomic usage
revert add godot version in backtrace message
fix getting lwin rwin key state on windows
m851 report only without parameters
assetlib update link for renamed repository
ensure sequence header is ok when parse avc aac data
force inspector to clean up on delete fixes crash
ae fixes for passthrough
fix a status return in replication handler
fix current time would go wonky when seeking while player is paused
bug fix compiling error
fix class memaccess warnings
place in anonymous namespace
update comment with note regarding items in a folder with content set that aren t yet in the db
less verbose logging when fetch is impossible
when importing video library from videodb xml make sure we import all the art specified
do not broadcast block if syncing with peer
astprinter print all spi attributes including the inherited ones
missed changes from previous checkin which stops freeing physical pages after a latest version logical page rewrite
fix a comment
better propertyeditor search
only force enable peripheral add ons with button mapping support
fix color region calculation for the first line
remove unused line
sema visit hoisted extensiondecls in extension binding
paplayer fix ff rw in cue file
clean up gdscript template
pvr fix remove context menu item add to favourites for children of repeating timers
clang format fixes
sema remove unnecessary logic for objc accessors
cupnpserver fix bad videodb url
fix last message tail drawing in theme preview
gardening remove written but never read variable found with clang tidy
merge remote tracking branch origin master into master next
default keep dim is true
importmulti options are optional
dbg fix visual bug in reference view init
merge branch master of github com mongodb mongo
pvr fix update addons get also disabled addons
use static internal attribute names instead of inplace strings
adsp add todo comments
use stringstream instead to to_string
add a separator in debug output for readability
fix do not derefence nullptrs log using id string
add miss api
revert timeline test
integral perf tests renamed for readability
oops missed porting over the line to set the alias decl end
mmalrenderer add auto select as explicit deinterlace method
omxplayer drop rest of groupid from omx player
update src interpreters databasecatalog cpp
changed these are const strings so declare them that way
resolved compiling errors under linux
remove unused variable
properly escape in strings when exporting csv
available space should take into account both memory and disk
gardening eliminate never read variable found with the static analyzer
remove license comments
detect usehd mismatches when wallet dat already exists
add space for readable
support label widget
wasapi do not probe for ae_fmt_aac we don t passthrough it anyways
fix warning about potentially unintialised variable
remove extra braces
fix tileset shape data not updating when being set via code
fix build error
fix no update path for pre frodo scrapped thumb url without aspect attribute
get hostname without mutex
remove fade in animation if no running animation
invoke codeformat on changes ingested from inbox
simpler debug output for deploy stow
add header comment
pvr pvrguiinfo fix playing times for addons not supporting getstreamtimes
no twisted mathematical hack
prevent lock from being moved from one thread to another
fix possible null pointer dereference in pvrmanager cpp
gui fix container bools
some minor fixes that got in the way of debugging a separate issue
dvdplayer do not pass display time to a v players if not initied
re add color lab test
declare integer constant ll
activeae recreate sink buffers if frame size has changed
silgen remove unnecessary isfinal check from getbaseaccessorfunctionref
make options hidden
implement horizontal scrolling with shift scroll this is standard scrolling behaviour in most applications
serialization temporary workaround to validate accessors when needed
convert lrn bias
pvr ignore very first server not reachable notification
compile error fix
imgproc fix removeoverlaps with empty input vector
omx report decoded image name
libexif add processing of comment tags used by windows
changes always show grid hotkey to prevent conflict with pan mode
use long_ptr with setwindowlongptr
fix bug introduced in merge commit
tweak context menu example
fix bug where new auths were not findable
add truncate for coordinator
remove an accidental mnist update
videoplayer dxva fix color values this fixes possible green border issue in case of decoder texture in pixel shaders
don t test map k v bucketcount since it s platform dependence
fix code informal
ci cover kernel codec
merge branch master of github com rhendz cosmos
fix bug for check having label or not for libsvm
fix bug of handshake with nginx simple hs
allow dashes in music videos on the first pass
fix mac build
update mergetreeindexconditionbloomfilter cpp
fix api_version and api_version_min numbers
switch to head_catchup instead of in_sync from lib_catchup
move disable_all_steppers to minkill
gardening fix recently introduced typo transistive transitive
sqrtf to std sqrt
fix crash when deleting a partially initialized memoryinstance after its address space allocation fails
omxplayervideo send player_avchange message when stereo mode changes
header comment add
fix dropdownbutton m_button style left align text and expansive widget
use caps log warning when using unknown http response codes
merge remote tracking branch origin master into master next
oops useless assignement
add comment and remove whitespace
fix asan report in extractall function
re enable gelf logging
don t break torrent checking
prevent division by zero
fix retroplayer audio after ae updates
make clearfailat always available again not only in maintainer mode
joysticks fix broken wheel and throttle input
disable hex mode after printing diagnostics
define add button as default in torrent addition dialog
fix warning deleting pointer to incomplete type
support hls callback with duration
fix bug show currect image must run app twice
fix mediacodec prevent crash if createbycodecname throws an exception omx rk
remove unused variable
restored try catch in the i o test
fix uimage constructor error
updated client error message for too many status requests co authored by tclinken trevorclinkenbeard gmail com
can pick how long sleep command sleeps for
add null ptr check in colour use
condence tracename parameter to trace
cmd mnemonic on macos now can press a button
merge remote tracking branch origin master into master rebranch
fix possible crash introduced by cvround
only hit cache once not twice
minor fieldrange cleaning verify on attempt to construct a fieldrange from expression ne which is unused and unsupported
guifontttf add missing offset to get limited values properly
fix include paths
fix call panel with empty user photo on retina
fix space issues
fix remove the debug loging
videoplayer rt streams flush demux queue if sync error is too big
findone throws better error
need to reset the accelerometer before trying to enable it again
fix compile error
fix for lazy init
lcd_map_control deps on ultipanel
fix issue with non empty existing blocks index file
minor change in customizedguimanager onprocessmessage
fix warning in socketio cpp
cv comparehist cv_comp_intersect
fix hls muxer segmentation fault
try to kill warning
fix webserver eating first char from header value
main tidy up enum comparison enum classes are comparable with one another so these casts aren t necessary
pidtempbed on endif
remove extra tab
changed might as well url encode the extension now that urlencode is working properly
merge remote tracking branch origin master into master next
std format fmt format
fix setlocale issue other than windows
add datetime conversion error message in backup
make sure tile is selected when selecting tilemap
fix possible crash with nullptr screen
fix error when trying to enter to folder node directly via address bar
fix miscellaneous warnings from compiling with mingw note that mingw defines nominmax by default
stdstring get rid of cstdstring in listproviders
fix prevent infinite loops in add on installer
little fix in command line parser catch unknown parameter types
fix flickering video osd
bug fixes for nacl port
another instance of downloading actor thumbs for back compat make sure we check the silly setting first
fix avoid crashing due to slow starting pvr manager
add shortcut to rename files
fix gles enable generic bob deinterlacing
fix undo redo scrollbar calulations
yet more compilation fixes
omxplayer seeks due to audio stream switch should be trickplay
move lcd_refresh_zprobe_zoffset definition
allow to use arrow keys with tool_move
sourcekit fix an unused variable warning
compact some macro substitutions
blockpolygon clip to image rectangle
fix wrong arg placeholder
video adds listitem properties ischapter isbookmark
increase epsilon for pass sanity check
try decompress fast
videoplayer drop deprecated ffmpeg paramter refcounted_frames
fix bsonobj ownership
ae oss is nearly not used on most platforms don t make vdrfan afraid
some cleanup ahead of bltouch support
fix missing double quote
remove blank line at eof
minor perf improvement
remove vertical alignment code style guidelines say we don t do this
sil remove unneeded includes
deallocating needs to unhook the interceptor or it may call methods on a deleted window
rs was not showing optime in replsetgetstatus for self
url auth checkboxes checked by default
fix build error apparently the function is not available on windows platforms
dont fail on method that does nothing
fontcache map is now static as we do not need external linkage for it
do not report torrent being checked as queued
stringutils add test for sortstringbyname
gardening change an if else block to use early exits instead
skiplistindex does not state that it could serve for array indexes
remove useless includes
move client creation and run under try to show exception message
file replace cautobuffer usage with auto_buffer fixes memory allocation check
support reading from directorymonitor
use ld instead of zd
fix another detail in agencycomm
try to fix gcc compilation error
sync render state
add check for invalid file descriptor in cddaripjob cpp
remove unintended debugging change
remove converting transform and step
fix msvc warning in tbb kmeans
remove unused uid field inside paralleltcinfo
fix silence warnings on osx size is signed int
initialize the platform pointer
qt group variables below initial if clauses in amountspinbox stepenabled
typechecker add a comment to isoverridebasedontype to explain why we return false when generic signatures differ
sort command will not be available in the beta version
remove needless conversion to unique pointer
videodb fix browsing tvshows genre studios with masterlock enabled
integrate vadimma changeresetrnn into master
gui double click in trace browser now follows in disassembly
keep binds for connected signals when changing node type
fix a line where integer data was treated as floating point data
remove assert that isn t always valid
more debugging for buildbot
add initializer instead of constant upon saving cntk models in onnx format
avoid a segfault on getblock if it can t read a block from disk
pvr cpvrchannelgroup searchandsetchannelicons remove some useless code
what s my name again
fix style error
nfc add some todo comments for cleanup
remove redundant new line
fix type for constant lowcardinality arguments
close read end of pipe to avoid leaking descriptors minor
better debugging for background issue
fix crash in theme applying
fix ccimage for msvc someone add std tolower which broke it
use ms restriction tag for windows store build
update valuesblockinputformat cpp
revert fix min size bug of label
remove dead code in filesystemdock
fix wrong viewport transparent bg
constraint solver fix incorrect bailout in connected components address a silly mistake that meant we were miscomputing connected components
remove smartquorum for coordinators check
davdirectory use charset reported by http server
drop raise_z_after_probing function
fix crash on unshuffle davilla says std sort requires a sort function that creates a strict weak ordering
fix getaddrinfo failing on android
modify the parsing
whitespace tabs to spaces
makes consistent slider joint gizmo
calculate lineedit selection with secret character
allow tls to proceed without node key if enrollment disabled
fix label autowrap clips text
gui file manager should always browse into file folders on clickk
fix inserting bezier curve in animation editor
try and increase fuzz coverage
prevent abort on shutdown if the fs is readonly
remove unnecessary trace line
merge remote tracking branch origin linux gcc into sls
changed error message in algorithm get
remove whitespace change
prevent to uncheck selected item with button group
overlays include padding in initial ass texture width computation
fix warning in console example we re iterating chars
don t show entries in listcommands for oldnames
remove unreachable code
applied guiscale to loaded skin
addongui fix fallback method if addon does not provide a skin file for current skin
aesinkpulse make clear what we open makes logfiles better readable
jsonrpc don t allow audio video playlist remove to remove the currently playing item
droid fix build
fix quit by closing non authed window
fix recruited the wrong number of log routers
fix decodebytes bug
cstaticlistprovider remove redundant check
minor text alignment
mmalcodec explicitly init variables
fix help display
remove useless semicolons
multiply album ratings
log block size and weight correctly
add required header the header was missing and our cmake ci builds failed as a result
updated help text
add is_inside_tree check in both timer and tween tween now throws error and doesnt even execute
fix invalid trace event detail named type
merge branch master of git github com mongodb mongo
suppress warning caused by ntstatus h inclusion
fix transactions created in fdbcli did not wait on the result of committing
remove some warnings
fix rest errors found by style check
add test case for std bad_alloc
fix grammar confusing sentence
use the unicode multiply sign for the a x b visual script node
update mergetreedata cpp
set project name as directory name instead of
don t close unless open
optimize rendering code
pvr cosmetic remove duplicate include of stringutils h
add check for allegro_init
export vector of float64 to python
core ppc add yield support
fix windows build was broken because statusschema string was too long
forgot to update the error message of add record
fix warning in gcc
fix for ignoring protocol options in cshoutcastfile open something like useragent referer etc
restore a hash check as an assert_we_think
add missing header for utility
add fragcoord to spatial light processing
add missing inequality test for crypto signature
fix rs compile
videoplayer fix bitrate calc for audio
gui fix copy selection
media library ignore trailing slashes when comparing the start directory path with the current directory path
test with logging disabled
remove extra declaration
pycaffe build include python first in caffe tool
remove debug output
revert removal of invalid interpolation check
merge branch slim into slim account resource window
nvdisp_disp0 fix broken assert
wallet improve reordertransactions
fix limit without order
remove deprecated methods
don t use deprecated qurl topleveldomain
fix possible socket leaks on error in cvtpsession getstreamlive
pvr fix deadlock in cpvrmanager checkparentallock when the manager is being stopped
fix lp based stabilizer
fix uninitialized member in paplayer cpp
oops wrong uassert nums
remove double space
aesinkdirectsound add dts to supported list as it s supported
cosmetics remove unneeded include and namespace from videoinfodownloader cpp
save trimmed string in preference
search modify message
generic specializer remove workarounds gsb now checks internally that generic signatures it produces are idempotent
fix ubsan and msan report in datelut
wallet remove unused threadflushwalletdb from removeprunedfunds
remove redundant islocked check in newkeypool
replace ostringstream with reusablestringstream in consolereporter
fix editor crash after closing scene with certain types of nodes
no minimum size for popups
don t change window flags in window mainwindow recountgeometryconstraints
remove commented codes
reduce severity of getdiskstatisticsdevicenotfound to sevwarn
chg mysql log server version
fix my website s url
eos_assert instead of check for null recovered_keys
remove wrong assert in torrent creator
remove finished todo comment
fix compile error
support hid_usage_page leds in dump_hid_value
fix compile issue
add sampling memory profiler
fix sometimes select plugin setting from context menu not works problem
remove sd file available size saturation
horizontal must be the default sprite sheet type as in old versions
fix videostreamgdnative audio buffer handling fix the audio buffer start when there are previous remains
skip ssh test
inject columns after creating header
bitcoin qt fix opening verify message tab via main menu
sharding object not array
fix cssrc putfloatdata wasn t rounding correctly
remove unused imports
set the show title for episodes during scan so that it s available in the announcemanager
astmangler enumelementdecl could also be a declcontext with a parameter list
combo recycling windows by using a stack number instead of a unique id wasitng less windows
fix rendering of active sprite when color management is disabled
now jstandard_theme regen does not use ji_get_theme
ast fix astdumper for argumentshuffleexpr
ast add a new verifier check for methods
change as suggested by original programmer and user of ecc software
fix invalid assertion
fix a warning
fix inf loop in jni error handler
add a label to the dock positioner
windows cleanup includes
make prepare_move_to_destination_dualx a proper move function
redraw all editors mainly the preview after flashing the layer
fix test data layer post lmdb
one another build fix
silence a compiler warning
remove debug logs
videoplayer fix pixel format for ffmpeg based hw decoders
small fix to avoid warnings
show n a for mined transactions and send to self instead of empty field
add notifier of archiving pinned dialog to update touchbar
enable copy elision
metrics windows display storage size
fix possible null dereference in guikeyboardfactory cpp
disable logging to stdout
remove unused codes
bitcoin wallet add a missing closing parenthesis in the help
net send post verack handshake messages at most once
serialization fix a warning
make shell use exec instead of invoke
close media viewer when showing file in folder
vdpau fix draining mixer
doc provide rationale for randomization in scheduling
remove unused variable
progress on mongo plugin
gsb suppress redundant diagnostics for circular protocol inheritance
kernel thread corrected a typo in an assert about the processor id
add binary read to test cpp
execsql use the correct statement length
charsetconverter stringcharsettoutf8 skip conversion if not needed
gui cosmetics fix indent in guislidercontrol cpp
simplified code slightly
fix a bug when using edge collections in traversal
update tracecollector cpp
make class cudnnsoftmaxlayertest derive from gpudevicetest
add missing test
buttontranslator add ability to disable lonpress mapping in sub windows
add precision scrolling to spatial editor zoom
sgbm fix uninitialized memory in cost buffer
fix typo in help text
workaround of sortkeys example
fix clang build one more time
fix updating of collision shape when the transform is set
silence a compiler warning in enterprise mode
fastrestore test generate random addprefix
clarify abortrescan rpc use
commented unused constants in the rho algorithm
center lines vertically in completion box
bench remove support
remove unnecessary include
silprinter be more tolerant if an instruction is not contained in a parent block
dependency scanning cope with inferred module maps
merge remote tracking branch origin master into master next
rbp omxplayer allow the use of external subtitles for dvds
format comment signed off by stefan weil sw weilnetz de
fix window close crash in jsb
sema fix lvalue ness constrain
merge remote tracking branch origin master into master next
update currentthread cpp
fixbug load obj file s material
better error message if tor version too old
revert recent memory fix for windows
make sure ptr is called with proper parameter otherwise the assertion fails
fix some compilation errors with gcc
qt do proper shutdown
merge remote branch bluematt dpifix
remove debug output
merge branch master of https github com godotengine godot
fix bus error in debugprint if datadir given non existent directory
videoplayer discard buffers instead of flush when hiding video
merge remote tracking branch origin master into master next
making the test deepper
move the previous window handling in login screen to onaction
dnn fix registration of custom opencvlayer do not require extensions library
util check for file being null in directorycommit
improve help msg
modify float convert to unsigned int
osx fix add a delay before audio renderer encoding init passthough and using adjust refresh rate
do not probe joypads if directinput cannot be initialized
fix formatting of settings clause of the insert query
fix do not enable the add button in media source dialog in case the first path is empty
dvdplayer consider audio stalled as long as message queue is empty
dbg better initialization messages
track cloned bytes
changed we should always check for multistream when caching
reduce number of stat syscalls for mergetree data parts
fix for timestamp style
disbale cache for s_attributeflags
fix typo in ibl computation comment
wow tried waiting for cluster members to shutdown forgot to set bool flag fail
activeae fix compiler warnings
add in check for agency precondition
remove the comments for clippingnodetest
simplify overengineered code
use fileno hpp from fc library conditionally output trailing n
fix some sema typos
pvr fix gui info time format for timeshift buffer start buffer end playing time
merge remote tracking branch origin master into master next
remove unused variable
update num threads in processors
small formatting fix
update range cpp
tighten hascuda check
fixes to build after merge
dvdplayer fix memleak in ffmpeg demuxer
fix compile error
slicelayer fix whitespace
dvdplayer adjust threshold of audio errors to frame duration
disabled net checking for dll
namedlazymemberloading avoid checking protocol existential conformances
xbmc url handle win lib path as a local resource
do not round input amounts allow rpc full precision
minor bug fixes
don t forget to strip reference storage types off
fix regression in core arithmetic
set _haspremultipliedalpha to false
don t shorten without need
optimize lsdive logic
add profileevent for dropped traces due to pipe full
kernel sessions clean up the list of pending request threads of a session when the client endpoint is closed
make it thread safe
apt fix typo in setting return code for notifytowait
typechecker nfc delete rudimentary autoclosurelistener isindefaultargcontext
keep undo history window position between sessions
wallet add regression test for vvalue sort order
context menu fix positioning
remove redundant code
update test cpp
test for failure
update python urls
clear font texture view to ensure release doesn t get called twice
prevent a buffer overflow in divetofile co authored by akunt
fix compile warning drop unused variable
throw exception when no secondary is valid and there s no master
fix invalid assertion
vaapi add check for maximum surface count in ensuresurfaces
skip channels without write access in forward box
merge remote tracking branch origin master into master next
fix an editorpropertyresource bug where clicking on the flat buttons would not select the property in the inspector
fix condition in keypointsfilter retainbest
aechannelinfo adjust the fix layouts to ffmpeg order
fix potential null pointer dereference in ccmain paragraphs cpp
fix missing includes
pvr changed add the filename of the recording if any to the epg tag serialised data
remove early cleanup
make sure abl manual starts close to the bed
nfs make use of the dns cache when resolving hostnames
need to fix tests first
set up authinfo for auto repair migration
minor oops move include
increase the probability to generate keys after xff xff to test special key framework code
joysticks fix all hell breaking loose when buttons are held
fix build break in no gpu cpp
fix wrong placement of new line in physicsbody warning
properly convert wchar to char before printing
restore setfamily in crash report window since setfamily in qapplication setfont was remove
fix bad test
merge branch master of github com apple foundationdb
first sd status change is silent
fix build error
if we have a null epg info tag continue the loop
winrenderer fix video after window resize
add some errors code for libcurl add https download support
fix hovering on toggled texture buttons fix hovering on toggled link buttons
don t make oplog very big on os x
videoplayer winrenderer fix uo sbs playback if stereo mode was enabled before renderer configure
brace style fix
ownership move ownership lowering past the eager specializer on the stdlib
remove stupid arduino bootloader can t handle this
disable clang format temp
buggify fastrestore_get_range_versions_expensive knob for true and false
add extra empty frames if it s necessary on a copy paste operation of layers between different sprites
add two basic performance tests
add log in ccfileutils fullpathforfilename for marmalade port
properly honor ctrl c
btree cpp free a btreebucket where we weren t
fix incorrect value for cc bit in iadd
fix function signature in error reporting
accept review suggestions co authored by a j beamon ajbeamon users noreply github com
show first message when not show version
android never use secure decoder for unencrypted material
a couple small tests in preparation
fix code checked in that didn t compile
show better tooltip for force muted in voice chat
fix loadblock after shutdown during ibd
a colon ending a comment line no longer indents the next line
set emission rate when settotalparticles
initialize the music and video thumbloaders in the recently add job to take advantage of database and art caching
sema remove unnecessary hack
displaying window size in debug mode
cs dump the number of viable solutions with debug constraints
reverted usage of vpp in cluster internal replication
abort when total assertions failed is greater than or equal to configured value
adds a comprehensive error message when jpeg loading fails
don t show symmetry controls on tools like eyedropper move zoom
fix build warning
jpegio remove unnecessary early return
svc updated outputdebugstring to use os_log
correct truncated comment
fix button control not updating value of label2 if set to
make c work for mongodump
bugfix use unique autostart filenames on linux for testnet regtest
replace auto with const auto in for loop
transform csgshape collision when necessary
fix bug commandlineparserparams s comparator would fail with equal arguments
always close next loaded pic if not matching slidenumber
make sure qb extensions are not shown in the web ui
backupworker remove assert in pop
fix wrong value for query contextownedbyexterior
video cvideoinfotag cleanup use std move in combination with by value parameters
merge branch master of https github com apple foundationdb into lazily fetch health metrics
remove some unused headers
keep groups when replacing nodes
merge remote tracking branch origin master into master next
fix issue where the properties panel data would not match the currently selected torrent
better version of sync checkpoint
avoid null pointer dereference when _walletmodel is null which is valid
more human readable svd code output
update context cpp
tolerate non empty braces in braces
update context cpp
reset method purge definition
one more fix
remove double logging of startup time when flogtimestamps
add block_id_notify as newest net_version
fix incorrect length calculation for keywords from iptc tags in which could result in a buffer overflow
ignoring space character whenever tesseract returns it
merge remote tracking branch origin master into master next
videorefclock fix potential segfault on systems which do not support video sync
hammer fix compilation
fix include casing
stitching perf increase threshold of transform vector
fix font setting reset on skin change
merge branch master of github com filipecaixeta clickhouse into filipecaixeta master
fix double free of astnode
use simpler ifdef style
add key generation verification to ecc sanity check
fix compile error
merge remote tracking branch origin master into master next
some compat tests minor
remove border from gamelist
pvr removes timer and guide item from root directory listing
adds attribute to non windows compilers
fix string concat wih first array argument in block
dbg show full command line near process started log entry
merge remote tracking branch origin master into master next
print help msg for shell at a more reasonable time during startup
add missing render format strings this should really be moved to some more generic location
add missing n in visual shader fresnel node code generation
fix inputeventkey echo type from int to bool
fix this control can t grab focus warning spam
add followerinfo to single server case so mop can test s
pvr remove double used header file
needed to add one more nullptr check on the kpe root type
fix crash just introduced
mmalrender skip some log message in bypass mode
ugly working code
fix exit code of watchdog
remove debug output
capture this instead of reference in lambda
fix dirty region rendering of containers when we don t necessarily update which item is focused while scrolling
pointers to pointers are very useful for tracking a node
add tegra stub for cornereigenvalsvecs
remove debug message
choose art don t show thumb were not applicable
resolution choose most matching refreshrate
interpolation was not remove because of id mismatch
shutdown stop threads before resetting ptrs
add note about non functional options attribute
attempt to fix some slots
activeae set samplerate before querying truehd
fix member keywords colour not updating on settings change
paplayer explicitely disable crossfading for audio cd
sil generic specializer canonicalize the substituted type of the specialized function in the context of a new generic signature
const argument in do_homing_move
fix weather manager not ready if settings get loaded for the first time
docs improve commenting in processgetdata
silgen add documentation to the fields of silgenapply
corrected initialization of smart pointer
fix don t add dxva surface to processor queue if the surface was requested to be dropped
fixup fix several merge issues
update pc show msg
update dns seed list
diagnostics fix contextualfailure to use solution data for hasappliedself
fdbmonitor fix some print scan format warnings signed off by austin seipp aseipp pobox com
fix for new use of callexpr create
chunk migrate pagefaultretryablesection for migrate insertion
update arrayreduceinranges cpp
size of the overlay should not be zoomed
add file extension to screenshot filename if not provided
fix compile error with tbb
remove printf in fop_to_save_document
fix don t pass musicdb url s to player
show amounts in b bold b in confirmation dialog
kernel use the correct format string for u64 hex
add an assert for defining serializer
fix the possible bug for key mapping in linux gtk library
fix faulty comment merge
fix build in xcode gcc
if the job manager isn t running don t add new jobs
make new tablebrowser class a bit more robust against misuse
allow choose art to work for seasons in the video info dialog
bugfix rpcwallet docs booleans aren t quoted
match data type of member variable
show delete action when multiple feeds are selected
fix unwanted flickering of record button when recorded data is empty
dbms src storages storagemergetree cpp
correctly report high s violations
more fuzzer fixes
merge remote tracking branch origin master into master next
paplayer handle prepare stream failure
code clean up
fix flipped scene preview thumbnail no need to flip the image generated from the viewport texture anymore
gardening remove unused variable ntd
solved merge conflict with readcache
full recently add update only after the db scan is finished
fix double delete
adjust guidialogvideoscan hide progress controls if not used
fix an issue with the way std chrono duration is defaulted with respoect to units in different compilers
add force option to fdbdr switch
reduced margin on popupmenu
don t use nullptr
fix textedit cursor column having a negative value
small arrangement in scrollcontainer
merge remote tracking branch origin master into master next
gardening re flow a small comment nfc
addonbrowser list all repositories on top regression after refactor
fix totals without having
animationtreeplayer constructor now sets processing mode
videoplayer dxva fix return value of decode
simplified the function sumofprimesupto
sema remove unreachable code from csdiag
more robust input box
cleanup settings remove dead code options filler epgguideviews
share included expression field name strings
fix scripting engine members initialization order
update prettyspaceblockoutputstream cpp
fix adam merge if right state wasn t initialized
cosmetic unneeded checks
large value for min max
small change to run ci
fix uninitialized memory usage in ann_mlp
remove invalid assert in appbrushes save
explicitly set clienttaskid
move ntimebestreceived updating into net processing code
remove redundant line
fix potential deadlock in player
so far so god
use std map emplace instead of std map insert
prevent shader crash if name of variable overrides function name
fix wrong property type for fix icon size in itemlist
make return type explicit
add debug output for multiline plates
remove incorrect transform preview from visual shaders
aa branch fix input cursor
correct a typo
fix distributed training hang with empty minibatch
updated pl u with more service names
revert syncing change
create openal context without attributes
capture resolver to keep it alive
add missing head
update replicatedmergetreecleanupthread cpp
better error message
prevent duplicate connections where one is by name and another by ip
add a missing switch break
show bones after creating
i fix type of execepton so it gets caught
builtins fix runaddon forwarding to runscript for weather subtitles lyrics scripts
add comments describing what the newly add function does
rfc1123time increase buffer size for time string make sure we can accomodate all possibilities
update assert number
trace record during trace
egl hook up font loading unloading on window init destroy
remove reference to unused variable
fix typo sentence starts with capital letter
optimize the algorithm to make the logic clearer
fix keyboard scrolling in contactsbox
fix gdnative procedures returning godot_char_string
getthreadid for freebsd
add missing space in storagedistributed logging sharding key
fix the upper boundary when calling checkrange thanks to alalek
gdscript allow accessing constants of outer classes
fix trailing n reporting extra text height
reset chosen filter on escape
change theme button name to edit theme ci skip
add initialize and generaterandomdata stubs
note on segment size
add the possibility to create bigger images to the limit of word size which is the ase format limit
scaling the minimum detector size defeats the purpose of resizing the image
remove extra multiplication in baker
better exceptions form event loop
update fibonacci cpp
inspectordock change label offset
visualization prevent null zero length packets from being processed
style police fix badly readable code in processmessage i introduced the problem and fix it with better readable code
databaseutils fix potential exception during sorting of musicvideos
fixes setting top_level not updating the global position
egl fix amlplayer build after egl rewrite
minor merge matcher changes
made sure the file priorities in torrent properties don t overwrite download first last pieces first option
fix compile error
constraintsystem defensive guard not to join unresolved type in potentialbindings addpotentialbinding
fix layout in the spellchecker settings block
really ignore the error from uploading actor
incrparse fix lexer offset issue when missing tokens get synthesized
merge branch develop of github com winlinvip simple rtmp server into develop
init fix help message for checkblockindex
use a void instead of an opaque sourcekitd_uid_t pointer in skuidtouidmap
update platemask cpp i just forgot this
fix compile error
fix compilation error on win
lua fix sprite newtag so it generates undo info and can be used in transactions
fix wrong compiler option
keep transition value when replacing key
pycaffe make pynet a class not a struct
another warning in valuetest
change to in conditions judge
merge branch master of https github com telegramdesktop tdesktop
allow directly read xff xff management without setting option
remove unused variable in getdirectory
cosmetic fix too few arguments for format
ensure csg parent s _make_dirty is called when entering a tree
better syncclusterconnection error reporting
formate the space
using correct config ref to initialize odbc bridge
core hle kernel clear process list on boot
changed get rid of the resume point query in the video info dialog we already know about it
change the background color of stage complete to yuzu blue
fix jsoncompact field delimiter
get rid of warning
add a compact on deletion collector to tables
delete two more lines
simplified cache eviction policy knob arg check
fixup extension printing
sharding fix startfrom
brace style fix
revert back test image
allow typing ctrl space
mono make sure editor thread is attached on script reload
header comment add
csapply make sure that hole type are not propagated to ast
addon transfer contentlookup properly
disable translation of createcont
review some review comments from aj
unittest match baseline
revert pre compute break statement
avoid shadowing variable in scope
armemu fix sxtab
curl udf the separator between the image path and the file path inside the image must be
fix bug running into infinite loop if distance between two adjacent points in curve are too short
supplement invoke destroyinstance of some reader in destroycocosstudio
fix hidden option handling
rbp omxplayer fix messanger handling
fix a typo in derivedconformances
nfc remove unneeded void cast of clangdc this is just a simple cleanup
update zookeeper cpp
add mkv to the list of video extensions
fix style check
bugfix gui use unsigned long long type to avoid implicit conversion of msb check
changed allow calling non plugin extension points as plugins
application show busy dialog when starting playback also for pvr recordings
cosmetic fix indentation
let const auto
ae properly cleanup on error
lowercase warning for buildbot minor
sql_db_plugin using const char for options
fix camera position
reactivated vertexcollectionrestrictions in neighbors
remove unused variable
nfc correcting minor typo on a csdiagnostics comment
cleanup the potential memory leak when the addon constructor fails
expose vector2 abs method to scripts
pregel fix concurrent creation of aggregator
drop an unused variable from parameter explosion mapping
sort data recovery sessions by date time
fix for undefined protocol
another useless include
particles2d fix initial size randomness property having no effect it was just a little typo p
timeline fix bug painting marching ants ouside the cel bounds
make uriutils getdirectory persinstent while calling inputstream open
paplayer wait for eof if no crossfading or cue sheet
fix editor crash when creating a custom resource from create dialog
aa branch fix
fix rare lock conflict on checkprocessmode
videoplayer fix audio stream not started
docs add missing backtick to option description it broke the startup option table generation
fix possible crash when changing the save path in torrent addition dialog
fix a bug in removeblock
fix issue resulting from the merge of scope changes
vcpkg move package abi access after they have been computed
add assertion to comparator
merge branch master of git github com mongodb mongo
update utils cpp
dvdplayer make sure not to display frames with pts pts at screen when rw
merge remote tracking branch origin master into master next
remove unused local variables this fixes some compiler warnings signed off by stefan weil sw weilnetz de
try to fix compile error
small update to async bench
add double support precentage of upower battery is a double value
indentation in marlinserial cpp
fix inactive cursors are still useful if their version is larger than the current version
fix warning when compiling in os x
fix compile for people with case sensitive filesystems
thread fix bug where result of __nextthread was not being properly checked when null
sema remove copyformalaccessfrom from destructor validation
fixes leak with copytofbshaderrd
use vscrollbar when calculating horiz offset in scrollcontainer someone forgot to change h_scroll to v_scroll when copy pasting the code
undo fix header for mergetreesequentialblockinputstream
proper readwritebufferfromhttp constructor args
improved ccd calculation algorithm
fix for buffer overflow problem with long filenames when setting up the leveldb
i e github requires a useragent thus enable it
bug fix starting a restore on a tag already in use would spinloop forever and eventually run out of memory
pass keyword inside a class implemented
report dictionary name on dictionary load errors
fix possible infinite loop with bucket fill tool
svc stub out svcgetcurrentprocessornumber
remove temporary file used for test
omg github cleanup merge stuff
dvdpalyer audio fix delivering the packet in wrong format
fix reading run lengths with empty values
include private header only if needed
vaapi set error if no video surfaces are available
add text to differentiate error messages
mmalcodec enable interpolate timestamps
ir visualizatin working
type checker ensure that we compute the objc bit as part of full checking
updated stack offsets for postmortem stack dump changed ets_putc to ets_uart_putc1 to better newline handling
fix warnings in ide
convertpointstohomogeneous fix output type for depth cv_32s
remove unused variable
fix two comments
correct usage of variable arguments
clean up web ui code
fix default separator
fix droid amc crash when calling dispose twice
log start and end of shutdown could be helpful when debugging shutdown related problems
fix absolute value
address book show unlabeled addresses as no label
make sure we enable implicit module maps
refine dvr donot append the flv header
internals fix calcsizecontents returning negative value on first run inconsequential afaik but fix for sanity
properly reset socket blocking state on close
minor changes in backupobserver backgroundthread
fix rate calculation typo
windowing wayland only build vaapi support if egl is present
fix droid feed codecinfo
changed camera interpolation to work when lmb is pressed and no modifiers are pressed
scroll history to down when sharing contact
check for global script class information before clearing it
reset default camera rotation
droid i think we have had enough of this spew
add block file naming helper blockfilepath
reorder xlinker and linker option group
parse fix lextrivia lf bug
disabled trap function back
bestgpu cpp changed lockgpu option to camel case
fix lint errors
check particlemodelrender to avoid crash
dragonflybsd thread renaming
use java audio for all devices while compatibility is tested
fix invalidation of the timeline range on kmousedownmessage
don t show receivedtill for local loaders
use index target in replace color for indexed images by default
moved datatypewithdictionary implementation to datatypewithdictionary cpp
update astalterquery cpp
add default value in timeout description
ae directsound use calculated sleep durations to reduce cpu usage
add back duplicate check
do not suggest the mutating keyword for getter subscripts
fix thumbs fanart would not show for video files with partial videoinfotag eg only resume point set
add running check
undotransaction does not need document wrappers
use deprecated pango_fc_font_get_glyph for old pango versions signed off by stefan weil sw weilnetz de
fix register physics2dtestmotionresult
add embedded art support for non musicdb items files
log python script name which was killed because it didn t stop when xbmc shuts down
don t show prefix if no elapsed time
moving location of variables to avoid conflicts between cn exe and dll building
fix compile errors in samples
check for interrupt in repair before we start real work
vaapi replace vaprofileh264baseline with constrainedbaseline baseline profile support in vaapi is deprecated and was never implemented in any driver
fix compile under linux
cguiwindowvideobase clear the cached source listing when remove a video source
fix vpack building bug
fix osx un fatfinger the repeat include
re add new lines at the end of files
guiinfo redact user pass in debug logging
update profileevents cpp
simplify code this avoids deferencing null pointer warning from static analyzers
wrong verbosity info on redirects
advancedsettings add proof and subs folders to movie excludefromscan expression
fix case of newseverity detail in stderrseverity trace event
pvr no need to assign a variable twice
add back forward declaration to address warnings
allow space between parameter and value
osx remove stray commented code
invoke fileutils getinstance getstringfromfile instead of fileutils getinstance getfiledata
clarify comment about the internal separation of read only and read write system roles
ncch only decompress code section
imagereader now understands imagelayout ids cudnn and legacy for mbformat
fix mediacodec handle aspect ratio
remove unused variable in loadfromxml
test activate segwit in testchain100setup
droid fix gl spectrum vis must have been busted by ae merge
cguilabelcontrol convert label to wide string to get correct string length
remove unused include
dont throw exception in v8 property getter
gif add more version ifdefs
clangimporter don t synthesize zero initializer body if we finished type checking
stitching fix l_gains data type from eigen solver float double
fix wrong branch
fix res error when dragging file to other file
normalizing vec from normalmap
dbg fix crashing
fix a bug in update_sliders_from_color
video osd remove unused include
fix duplicatied assertion number
core cache dir version update
eventclient fix wiimote battery problem eventclient fix wiimote battery problem remove commented code
merge remote tracking branch origin master into master next
final fix for the unit test case
profiles cleanup includes
outputting the number of cycles per input byte
attempt to fix weird index issue
resume if plugins set the resumetime in setresolvedurl then assume they want the item to resume
merge branch master of github com mongodb mongo
assert if we attempt to bind a type involving errortype
updated gui photoreceptors sensitivity paramaer management
remove unused variables in ocl_perf_test_p
fix to read advanced setting for artist art after coverity scan fixes
minor change to avoid compiler error
add a test for grouped deconvolution from onnx
don t parse the action in eventserver builtins want it unmodified
pvr fix string ressource comment
remove absolute path lookup in fullpathforfilename
runtime remove unused once_flag
fix include in coprocessor cpp
dvdvideocodecimx fix double rate deinterlacing
offset particle drawing order so that recently emitted particles are always drawn on top of the older ones
backup workers only save committed mutations
usr also add nominal check for the inheritance
fix previously introduced error when calling contract apply function
fix add optional parameter to python interface for executing builtin functions blocking defaults to non blocking for compatibility
print on error
fix crash using spritesizecommand on batch mode
un fix one of the warnings
fix logic errors
fix a couple of linux build issues
txindex remove unused boost thread
send onsettingschanged from caddon savesettings it works when changing settings from python as well thx pieh
peripherals don t overwrite device name if no name is found in peripherals xml
fix a deadlock
take some couts out to reproduce bug
dvdplayer remove resume players on closefile fixes some small portion of audio being played when stopping a paused video
fix problems in color curve effect using color indexes out of range
fix bug of forward erase msgs when error
just playing inserted a space
don t log first operation which is now coming from a dbdirectclient
don t use lcd alert unless error or alert level is reset later
fonts remove commented logging statement
fix problem deleting app document that are inside a context
don t change the hand cursor for styledbutton if the widget is disabled
configuration file changed from ase x x cfg to aseprite x x cfg
forced no duplication of animation data
add missing header comment
ifdef for readline version
add empty line
remove const test cases all tests passed
apply noless nomore
cosmetic changes to the article previewx tree
popups child popups menus hidden on their first frame the same way as regular popup
fix bug using uninitialized prev_x and prev_y values in editor editor_click
fix droid typos
remove unneeded lambda capture in abi serializer
add a cast to make windwos happier
smshell query array access mode
dbg fix a bug in debugupdategui dont update gui when requested address is invalid
fix crash when user pins chat in remotely remove filter
pictures remove unneeded out of bounds check
fix comments on startversion in backupworker
update partlog cpp
am implement iselfcontroller exitlock commands
ast simplify classancestryflagsrequest
